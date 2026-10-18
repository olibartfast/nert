// ConvTranspose on CUDA via cuDNN: NCHW float32 transposed convolution.
//
// API choice (N1b Conv precedent): the legacy cudnnConvolutionBackwardData
// path (descriptors + v7 heuristics + first-party bias add). The transpose
// Y(N,M,Ho,Wo) = ConvTranspose(X(N,C,H,W), W(C,M/group,kH,kW)) is exactly
// the convolution input-gradient with dy=X, filter=W, dx=Y. TF32 is
// forbidden (CUDNN_FMA_MATH; DEFAULT_MATH still allows TF32 on Ampere+)
// so results match the float32 CPU oracle
// within tolerance. Rank-4 only, explicit pads or VALID (the loader
// rejects the rest, like the oracle).
//
// Narrower than the CPU oracle, by cuDNN necessity (named rejections):
// non-zero output_padding (cuDNN sizes dx from dy alone) and asymmetric
// pads (cuDNN takes one pad per axis) are rejected here.
//
// Engine choice is cached per (dims, attrs) key: v7 PREFER_FASTEST
// heuristics first, workspace allocated once per key.

#include "nert/device.hpp"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace nert {
namespace cuda_kernels {
namespace {

void CheckCuda(cudaError_t status, const char* what, const std::string& context) {
    if (status != cudaSuccess) {
        throw InferenceException(context + ": " + what + " failed: " + cudaGetErrorString(status));
    }
}

void CheckCudnn(cudnnStatus_t status, const char* what, const std::string& context) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw InferenceException(context + ": " + what + " failed: " + cudnnGetErrorString(status));
    }
}

int64_t ElementCount(const std::vector<int64_t>& dims, const std::string& context) {
    int64_t count = 1;
    for (int64_t dim : dims) {
        if (dim <= 0) {
            throw InferenceException(context + ": non-positive dimension");
        }
        count *= dim;
    }
    return count;
}

int IntOrThrow(int64_t value, const char* what, const std::string& context) {
    if (value > std::numeric_limits<int>::max()) {
        throw InferenceException(context + ": " + what + " exceeds the cuDNN dimension limit");
    }
    return static_cast<int>(value);
}

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("ConvTranspose: attribute '" + name + "' must be an integer (node '" +
                                         node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

std::vector<int64_t> IntListAttribute(const Node& node, const std::string& name) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::vector<int64_t>* value = std::get_if<std::vector<int64_t>>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("ConvTranspose: attribute '" + name + "' must be an int list (node '" +
                                         node.name + "')");
            }
            return *value;
        }
    }
    return {};
}

std::string StringAttribute(const Node& node, const std::string& name, const std::string& fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::string* value = std::get_if<std::string>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("ConvTranspose: attribute '" + name + "' must be a string (node '" +
                                         node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

// y[n*M*S + m*S + s] += bias[m]: the rank-1 [M] bias cuDNN leaves out.
__global__ void TransposeBiasAddKernel(float* y, const float* bias, int64_t batch, int64_t channels, int64_t spatial) {
    const int64_t count = batch * channels * spatial;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        y[i] += bias[(i / spatial) % channels];
    }
}

// Cached engine choice + descriptors + workspace per resolved geometry. The
// handle and these objects are process-lifetime (single-threaded executor
// use, like the tables); the map intentionally does not clean up at exit.
struct BwdPlanCacheEntry {
    bool ready = false;
    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    size_t workspace_bytes = 0;
    void* workspace = nullptr;
    cudnnTensorDescriptor_t dy_desc = nullptr;
    cudnnTensorDescriptor_t dx_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
};

std::map<std::string, BwdPlanCacheEntry>& BwdPlanCache() {
    static std::map<std::string, BwdPlanCacheEntry> cache;
    return cache;
}

std::mutex& BwdPlanCacheMutex() {
    static std::mutex mutex;
    return mutex;
}

// One process-lifetime cuDNN handle (Conv's N1e lesson: creating a handle
// per call dominates). Shared with Conv's handle type but a separate
// instance owned here.
cudnnHandle_t TransposeSharedHandle() {
    static cudnnHandle_t handle = [] {
        cudnnHandle_t created = nullptr;
        CheckCudnn(cudnnCreate(&created), "cudnnCreate", "ConvTranspose");
        return created;
    }();
    return handle;
}

} // namespace

void ConvTranspose(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ConvTranspose (node '" + node.name + "')";
    if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected input, weight, optional bias, and one output");
    }
    const TensorView& x = inputs[0];
    const TensorView& w = inputs[1];
    const TensorView* bias = inputs.size() == 3 && inputs[2].data != nullptr ? &inputs[2] : nullptr;
    const TensorView& out = outputs[0];
    if (x.dtype != DataType::Float32 || w.dtype != DataType::Float32 || out.dtype != DataType::Float32 ||
        (bias != nullptr && bias->dtype != DataType::Float32)) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (x.dims.size() != 4 || w.dims.size() != 4) {
        throw InferenceException(context + ": input and weight must be rank 4 (NCHW)");
    }
    (void)ElementCount(x.dims, context);
    (void)ElementCount(w.dims, context);
    (void)ElementCount(out.dims, context);

    const int64_t group = IntAttribute(node, "group", 1);
    const int64_t nBatch = x.dims[0];
    const int64_t inChannels = x.dims[1];
    const int64_t inHeight = x.dims[2];
    const int64_t inWidth = x.dims[3];
    if (group <= 0 || inChannels % group != 0 || w.dims[0] != inChannels) {
        throw InferenceException(context + ": weight channels do not match the input and group");
    }
    const int64_t outPerGroup = w.dims[1];
    const int64_t outChannels = outPerGroup * group;
    const int64_t kH = w.dims[2];
    const int64_t kW = w.dims[3];

    std::vector<int64_t> strides = IntListAttribute(node, "strides");
    std::vector<int64_t> dilations = IntListAttribute(node, "dilations");
    std::vector<int64_t> outputPadding = IntListAttribute(node, "output_padding");
    std::vector<int64_t> pads = IntListAttribute(node, "pads");
    if (strides.empty()) {
        strides = {1, 1};
    }
    if (dilations.empty()) {
        dilations = {1, 1};
    }
    if (outputPadding.empty()) {
        outputPadding = {0, 0};
    }
    if (pads.empty() || StringAttribute(node, "auto_pad", "NOTSET") == "VALID") {
        pads = {0, 0, 0, 0};
    }
    if (strides.size() != 2 || dilations.size() != 2 || outputPadding.size() != 2 || pads.size() != 4) {
        throw InferenceException(context + ": strides, dilations, output_padding, and pads must be 2-D");
    }
    if (strides[0] <= 0 || strides[1] <= 0 || dilations[0] <= 0 || dilations[1] <= 0 || kH <= 0 || kW <= 0) {
        throw InferenceException(context + ": strides, dilations, and kernel dimensions must be positive");
    }
    if (outputPadding[0] != 0 || outputPadding[1] != 0) {
        throw InferenceException(context + ": non-zero output_padding is not supported on CUDA here");
    }
    if (pads[0] != pads[2] || pads[1] != pads[3]) {
        throw InferenceException(context + ": asymmetric padding is not supported on CUDA here");
    }

    const int64_t outHeight =
        strides[0] * (inHeight - 1) + outputPadding[0] + dilations[0] * (kH - 1) + 1 - pads[0] - pads[2];
    const int64_t outWidth =
        strides[1] * (inWidth - 1) + outputPadding[1] + dilations[1] * (kW - 1) + 1 - pads[1] - pads[3];
    const std::vector<int64_t> expected = {nBatch, outChannels, outHeight, outWidth};
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the transposed window");
    }
    if (bias != nullptr && (bias->dims.size() != 1 || bias->dims[0] != outChannels)) {
        throw InferenceException(context + ": bias must be rank-1 of length M");
    }
    if (x.data == nullptr || w.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    // Backward-data mapping: dyDesc = X [N,C,H,W], wDesc = W [C,M/group,kH,kW],
    // dxDesc = Y [N,M,Ho,Wo].
    std::vector<int> dyDims = {IntOrThrow(nBatch, "batch", context), IntOrThrow(inChannels, "channels", context),
                               IntOrThrow(inHeight, "input height", context),
                               IntOrThrow(inWidth, "input width", context)};
    std::vector<int> wDims = {IntOrThrow(inChannels, "channels", context), IntOrThrow(outPerGroup, "channels", context),
                              IntOrThrow(kH, "kernel height", context), IntOrThrow(kW, "kernel width", context)};
    std::vector<int> dxDims = {IntOrThrow(nBatch, "batch", context), IntOrThrow(outChannels, "channels", context),
                               IntOrThrow(outHeight, "output height", context),
                               IntOrThrow(outWidth, "output width", context)};
    std::vector<int> pad = {IntOrThrow(pads[0], "pad", context), IntOrThrow(pads[1], "pad", context)};
    std::vector<int> stride = {IntOrThrow(strides[0], "stride", context), IntOrThrow(strides[1], "stride", context)};
    std::vector<int> dilation = {IntOrThrow(dilations[0], "dilation", context),
                                 IntOrThrow(dilations[1], "dilation", context)};
    const int nbDims = 4;
    auto contiguous_strides = [](const std::vector<int>& dims) {
        std::vector<int> strides(dims.size(), 1);
        for (std::size_t i = dims.size(); i-- > 1;) {
            strides[i - 1] = strides[i] * dims[i];
        }
        return strides;
    };

    cudnnHandle_t handle = TransposeSharedHandle();

    std::string key = "ct";
    for (int d : dyDims) {
        key += std::to_string(d) + ",";
    }
    key += "|";
    for (int d : wDims) {
        key += std::to_string(d) + ",";
    }
    key += "|g" + std::to_string(group) + "s" + std::to_string(stride[0]) + "," + std::to_string(stride[1]) + "d" +
           std::to_string(dilation[0]) + "," + std::to_string(dilation[1]) + "p" + std::to_string(pad[0]) + "," +
           std::to_string(pad[1]);
    BwdPlanCacheEntry* plan = nullptr;
    {
        std::lock_guard<std::mutex> lock(BwdPlanCacheMutex());
        plan = &BwdPlanCache()[key];
    }
    if (!plan->ready) {
        std::lock_guard<std::mutex> lock(BwdPlanCacheMutex());
        if (!plan->ready) {
            const std::vector<int> dyStrides = contiguous_strides(dyDims);
            const std::vector<int> dxStrides = contiguous_strides(dxDims);
            CheckCudnn(cudnnCreateTensorDescriptor(&plan->dy_desc), "cudnnCreateTensorDescriptor", context);
            CheckCudnn(cudnnCreateTensorDescriptor(&plan->dx_desc), "cudnnCreateTensorDescriptor", context);
            CheckCudnn(cudnnCreateFilterDescriptor(&plan->w_desc), "cudnnCreateFilterDescriptor", context);
            CheckCudnn(cudnnCreateConvolutionDescriptor(&plan->conv_desc), "cudnnCreateConvolutionDescriptor", context);
            CheckCudnn(
                cudnnSetTensorNdDescriptor(plan->dy_desc, CUDNN_DATA_FLOAT, nbDims, dyDims.data(), dyStrides.data()),
                "cudnnSetTensorNdDescriptor", context);
            CheckCudnn(
                cudnnSetTensorNdDescriptor(plan->dx_desc, CUDNN_DATA_FLOAT, nbDims, dxDims.data(), dxStrides.data()),
                "cudnnSetTensorNdDescriptor", context);
            CheckCudnn(
                cudnnSetFilterNdDescriptor(plan->w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, nbDims, wDims.data()),
                "cudnnSetFilterNdDescriptor", context);
            CheckCudnn(cudnnSetConvolutionNdDescriptor(plan->conv_desc, 2, pad.data(), stride.data(), dilation.data(),
                                                       CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                       "cudnnSetConvolutionNdDescriptor", context);
            CheckCudnn(cudnnSetConvolutionMathType(plan->conv_desc, CUDNN_FMA_MATH), "cudnnSetConvolutionMathType",
                       context);
            CheckCudnn(cudnnSetConvolutionGroupCount(plan->conv_desc, IntOrThrow(group, "group", context)),
                       "cudnnSetConvolutionGroupCount", context);

            int returned = 0;
            cudnnConvolutionBwdDataAlgoPerf_t perf{};
            CheckCudnn(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, plan->w_desc, plan->dy_desc, plan->conv_desc,
                                                                   plan->dx_desc, 1, &returned, &perf),
                       "cudnnGetConvolutionBackwardDataAlgorithm_v7", context);
            if (returned < 1 || perf.status != CUDNN_STATUS_SUCCESS) {
                throw InferenceException(context + ": cuDNN found no backward-data algorithm here");
            }
            plan->algo = perf.algo;
            CheckCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, plan->w_desc, plan->dy_desc,
                                                                    plan->conv_desc, plan->dx_desc, plan->algo,
                                                                    &plan->workspace_bytes),
                       "cudnnGetConvolutionBackwardDataWorkspaceSize", context);
            if (plan->workspace_bytes > 0) {
                CheckCuda(cudaMalloc(&plan->workspace, plan->workspace_bytes), "cudaMalloc", context);
            }
            plan->ready = true;
        }
    }

    const float alpha = 1.0F, beta = 0.0F;
    CheckCudnn(cudnnConvolutionBackwardData(handle, &alpha, plan->w_desc, w.data, plan->dy_desc, x.data,
                                            plan->conv_desc, plan->algo, plan->workspace, plan->workspace_bytes, &beta,
                                            plan->dx_desc, out.data),
               "cudnnConvolutionBackwardData", context);

    if (bias != nullptr) {
        const float* bData = static_cast<const float*>(bias->data);
        const int64_t spatialCount = outHeight * outWidth;
        const int64_t count = nBatch * outChannels * spatialCount;
        const int block = 256;
        const int grid = static_cast<int>((count + block - 1) / block);
        TransposeBiasAddKernel<<<grid, block>>>(static_cast<float*>(out.data), bData, nBatch, outChannels,
                                                spatialCount);
        CheckCuda(cudaGetLastError(), "TransposeBiasAddKernel launch", context);
    }
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace cuda_kernels
} // namespace nert
