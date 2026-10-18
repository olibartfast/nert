// Conv on CUDA via cuDNN: NCHW float32 cross-correlation with groups,
// strides, dilations, explicit or automatic padding, and an optional
// rank-1 bias.
//
// API choice (recorded deviation from the N1b packet's v9-graph sketch):
// the legacy cudnnConvolutionForward path (descriptors + v7 heuristics +
// cudnnAddTensor-style bias done first-party) instead of the v9
// operation-graph API. Same library kernels underneath, far smaller API
// surface to get exactly right, identical numerics goal. TF32 is forbidden
// (CUDNN_FMA_MATH; DEFAULT_MATH still allows TF32 on Ampere+) so results
// match the float32 CPU oracle within
// tolerance. Rank 3 lifts to 4D (H=1 shares the buffer: the flat layout is
// identical); rank 5 runs as native 3D (NCDHW).
//
// Engine choice is cached per (dims, attrs) key: v7 PREFER_FASTEST
// heuristics first, workspace allocated once per key. Per-call algo search
// and per-call workspace malloc are the documented follow-ups, not this
// pilot.

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
                throw InferenceException("Conv: attribute '" + name + "' must be an integer (node '" + node.name +
                                         "')");
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
                throw InferenceException("Conv: attribute '" + name + "' must be an int list (node '" + node.name +
                                         "')");
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
                throw InferenceException("Conv: attribute '" + name + "' must be a string (node '" + node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

// Same formulas as the CPU oracle's ResolvePads (and shape inference), so
// the three never disagree.
void ResolvePads(const std::string& context, const std::string& auto_pad, const std::vector<int64_t>& in_spatial,
                 const std::vector<int64_t>& kernel, const std::vector<int64_t>& dilations,
                 const std::vector<int64_t>& strides, const std::vector<int64_t>& pads_attr,
                 std::vector<int64_t>& pad_begin, std::vector<int64_t>& pad_end) {
    const std::size_t rank = in_spatial.size();
    pad_begin.assign(rank, 0);
    pad_end.assign(rank, 0);
    if (auto_pad == "VALID") {
        return;
    }
    if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        for (std::size_t i = 0; i < rank; ++i) {
            const int64_t out = (in_spatial[i] + strides[i] - 1) / strides[i];
            const int64_t effective = dilations[i] * (kernel[i] - 1) + 1;
            int64_t total = (out - 1) * strides[i] + effective - in_spatial[i];
            if (total < 0) {
                total = 0;
            }
            if (auto_pad == "SAME_UPPER") {
                pad_begin[i] = total / 2;
                pad_end[i] = total - pad_begin[i];
            } else {
                pad_end[i] = total / 2;
                pad_begin[i] = total - pad_end[i];
            }
        }
        return;
    }
    if (pads_attr.empty()) {
        return;
    }
    if (pads_attr.size() != 2 * rank) {
        throw InferenceException(context + ": pads must have two entries per spatial dimension");
    }
    for (std::size_t i = 0; i < rank; ++i) {
        pad_begin[i] = pads_attr[i];
        pad_end[i] = pads_attr[rank + i];
    }
}

// y[n*M*S + m*S + s] += bias[m]: the rank-1 [M] bias cuDNN forward leaves out.
__global__ void BiasAddKernel(float* y, const float* bias, int64_t batch, int64_t channels, int64_t spatial) {
    const int64_t count = batch * channels * spatial;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        y[i] += bias[(i / spatial) % channels];
    }
}

// Cached engine choice + descriptors + workspace per resolved geometry. The
// handle and these objects are process-lifetime (single-threaded executor
// use, like the tables); the map intentionally does not clean up at exit.
struct PlanCacheEntry {
    bool ready = false;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspace_bytes = 0;
    void* workspace = nullptr;
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
};

std::map<std::string, PlanCacheEntry>& PlanCache() {
    static std::map<std::string, PlanCacheEntry> cache;
    return cache;
}

std::mutex& PlanCacheMutex() {
    static std::mutex mutex;
    return mutex;
}

// One process-lifetime cuDNN handle: creating a handle per call was most of
// the measured Conv overhead (N1e slice A).
cudnnHandle_t SharedHandle() {
    static cudnnHandle_t handle = [] {
        cudnnHandle_t created = nullptr;
        CheckCudnn(cudnnCreate(&created), "cudnnCreate", "Conv");
        return created;
    }();
    return handle;
}

} // namespace

void Conv(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Conv (node '" + node.name + "')";
    if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two or three inputs and one output");
    }
    const TensorView& x = inputs[0];
    const TensorView& w = inputs[1];
    const TensorView& out = outputs[0];
    const bool hasBias = inputs.size() == 3;
    if (x.dtype != DataType::Float32 || w.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (hasBias && inputs[2].dtype != DataType::Float32) {
        throw InferenceException(context + ": bias tensor must be float32");
    }
    const std::size_t rank = x.dims.size();
    if (rank < 3 || rank > 5) {
        throw InferenceException(context + ": input must be rank 3, 4, or 5 (1-D, 2-D, or 3-D convolution)");
    }
    if (w.dims.size() != rank) {
        throw InferenceException(context + ": weight rank must match the input rank");
    }
    (void)ElementCount(x.dims, context);
    (void)ElementCount(w.dims, context);
    (void)ElementCount(out.dims, context);
    const std::size_t spatial = rank - 2;

    const int64_t n_batch = x.dims[0];
    const int64_t inChannels = x.dims[1];
    const int64_t outChannels = w.dims[0];

    const int64_t group = IntAttribute(node, "group", 1);
    if (group <= 0) {
        throw InferenceException(context + ": group must be positive");
    }
    if (inChannels % group != 0) {
        throw InferenceException(context + ": input channels are not divisible by group");
    }
    if (w.dims[1] != inChannels / group) {
        throw InferenceException(context + ": weight channel dimension does not match input/group");
    }
    if (outChannels % group != 0) {
        throw InferenceException(context + ": output channels are not divisible by group");
    }

    std::vector<int64_t> strides = IntListAttribute(node, "strides");
    std::vector<int64_t> dilations = IntListAttribute(node, "dilations");
    if (strides.empty()) {
        strides.assign(spatial, 1);
    }
    if (dilations.empty()) {
        dilations.assign(spatial, 1);
    }
    if (strides.size() != spatial || dilations.size() != spatial) {
        throw InferenceException(context + ": stride and dilation must have one entry per spatial dimension");
    }
    const std::vector<int64_t> in_spatial(x.dims.begin() + 2, x.dims.end());
    const std::vector<int64_t> kernel(w.dims.begin() + 2, w.dims.end());
    for (std::size_t i = 0; i < spatial; ++i) {
        if (strides[i] <= 0 || dilations[i] <= 0) {
            throw InferenceException(context + ": stride and dilation must be positive");
        }
        if (kernel[i] <= 0) {
            throw InferenceException(context + ": kernel dimensions must be positive");
        }
    }

    std::vector<int64_t> pad_begin;
    std::vector<int64_t> pad_end;
    ResolvePads(context, StringAttribute(node, "auto_pad", "NOTSET"), in_spatial, kernel, dilations, strides,
                IntListAttribute(node, "pads"), pad_begin, pad_end);

    const float* bias = nullptr;
    if (hasBias) {
        const std::vector<int64_t>& biasDims = inputs[2].dims;
        if (biasDims.size() != 1 || biasDims[0] != outChannels) {
            throw InferenceException(context + ": bias must be rank-1 of length M");
        }
        bias = static_cast<const float*>(inputs[2].data);
    }
    if (x.data == nullptr || w.data == nullptr || out.data == nullptr || (hasBias && bias == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    // Lift rank-3 to 4D (H=1 shares the buffer: the flat layout is identical);
    // rank-5 runs native 3D.
    const std::size_t cudnn_spatial = spatial == 1 ? 2 : spatial;
    auto lifted = [&](const std::vector<int64_t>& v, int64_t fill) {
        std::vector<int64_t> r(cudnn_spatial - spatial, fill);
        r.insert(r.end(), v.begin(), v.end());
        return r;
    };
    const std::vector<int64_t> inS = lifted(in_spatial, 1);
    const std::vector<int64_t> kS = lifted(kernel, 1);
    const std::vector<int64_t> sS = lifted(strides, 1);
    const std::vector<int64_t> dS = lifted(dilations, 1);
    const std::vector<int64_t> pS = lifted(pad_begin, 0);
    const std::vector<int64_t> eS = lifted(pad_end, 0);

    // Output spatial dims from the oracle formula (floor, no ceil mode).
    std::vector<int64_t> oS(cudnn_spatial, 0);
    for (std::size_t i = 0; i < cudnn_spatial; ++i) {
        const int64_t effective = dS[i] * (kS[i] - 1) + 1;
        oS[i] = (inS[i] + pS[i] + eS[i] - effective) / sS[i] + 1;
    }
    std::vector<int64_t> expected = {n_batch, outChannels};
    expected.insert(expected.end(), spatial == 1 ? oS.begin() + 1 : oS.begin(), oS.end());
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the convolution");
    }

    // Descriptor dims: [N,C,...] / [M,Cg,...] / [N,M,...] with the lift.
    std::vector<int> xDims = {IntOrThrow(n_batch, "batch", context), IntOrThrow(inChannels, "channels", context)};
    std::vector<int> wDims = {IntOrThrow(outChannels, "channels", context),
                              IntOrThrow(inChannels / group, "channels", context)};
    std::vector<int> yDims = {IntOrThrow(n_batch, "batch", context), IntOrThrow(outChannels, "channels", context)};
    std::vector<int> pad(cudnn_spatial, 0), stride(cudnn_spatial, 0), dilation(cudnn_spatial, 0);
    for (std::size_t i = 0; i < cudnn_spatial; ++i) {
        xDims.push_back(IntOrThrow(inS[i], "input dim", context));
        wDims.push_back(IntOrThrow(kS[i], "kernel dim", context));
        yDims.push_back(IntOrThrow(oS[i], "output dim", context));
        if (pS[i] != eS[i]) {
            throw InferenceException(context + ": asymmetric padding is not supported on CUDA here");
        }
        pad[i] = IntOrThrow(pS[i], "pad", context);
        stride[i] = IntOrThrow(sS[i], "stride", context);
        dilation[i] = IntOrThrow(dS[i], "dilation", context);
    }
    const int nbDims = static_cast<int>(xDims.size());
    auto contiguous_strides = [](const std::vector<int>& dims) {
        std::vector<int> strides(dims.size(), 1);
        for (std::size_t i = dims.size(); i-- > 1;) {
            strides[i - 1] = strides[i] * dims[i];
        }
        return strides;
    };

    cudnnHandle_t handle = SharedHandle();

    // Plan cache: descriptors + v7 PREFER_FASTEST engine per resolved
    // geometry; workspace allocated once per key and kept for the process.
    std::string key;
    for (int d : xDims) {
        key += std::to_string(d) + ",";
    }
    key += "|";
    for (int d : wDims) {
        key += std::to_string(d) + ",";
    }
    key += "|g" + std::to_string(group) + "s";
    for (std::size_t i = 0; i < cudnn_spatial; ++i) {
        key += std::to_string(stride[i]) + "," + std::to_string(dilation[i]) + "," + std::to_string(pad[i]) + ";";
    }
    PlanCacheEntry* plan = nullptr;
    {
        std::lock_guard<std::mutex> lock(PlanCacheMutex());
        plan = &PlanCache()[key];
    }
    if (!plan->ready) {
        std::lock_guard<std::mutex> lock(PlanCacheMutex());
        if (!plan->ready) {
            const std::vector<int> xStrides = contiguous_strides(xDims);
            const std::vector<int> yStrides = contiguous_strides(yDims);
            CheckCudnn(cudnnCreateTensorDescriptor(&plan->x_desc), "cudnnCreateTensorDescriptor", context);
            CheckCudnn(cudnnCreateTensorDescriptor(&plan->y_desc), "cudnnCreateTensorDescriptor", context);
            CheckCudnn(cudnnCreateFilterDescriptor(&plan->w_desc), "cudnnCreateFilterDescriptor", context);
            CheckCudnn(cudnnCreateConvolutionDescriptor(&plan->conv_desc), "cudnnCreateConvolutionDescriptor", context);
            CheckCudnn(
                cudnnSetTensorNdDescriptor(plan->x_desc, CUDNN_DATA_FLOAT, nbDims, xDims.data(), xStrides.data()),
                "cudnnSetTensorNdDescriptor", context);
            CheckCudnn(
                cudnnSetTensorNdDescriptor(plan->y_desc, CUDNN_DATA_FLOAT, nbDims, yDims.data(), yStrides.data()),
                "cudnnSetTensorNdDescriptor", context);
            CheckCudnn(
                cudnnSetFilterNdDescriptor(plan->w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, nbDims, wDims.data()),
                "cudnnSetFilterNdDescriptor", context);
            CheckCudnn(cudnnSetConvolutionNdDescriptor(plan->conv_desc, static_cast<int>(cudnn_spatial), pad.data(),
                                                       stride.data(), dilation.data(), CUDNN_CROSS_CORRELATION,
                                                       CUDNN_DATA_FLOAT),
                       "cudnnSetConvolutionNdDescriptor", context);
            CheckCudnn(cudnnSetConvolutionMathType(plan->conv_desc, CUDNN_FMA_MATH), "cudnnSetConvolutionMathType",
                       context);
            CheckCudnn(cudnnSetConvolutionGroupCount(plan->conv_desc, IntOrThrow(group, "group", context)),
                       "cudnnSetConvolutionGroupCount", context);

            int returned = 0;
            cudnnConvolutionFwdAlgoPerf_t perf{};
            CheckCudnn(cudnnGetConvolutionForwardAlgorithm_v7(handle, plan->x_desc, plan->w_desc, plan->conv_desc,
                                                              plan->y_desc, 1, &returned, &perf),
                       "cudnnGetConvolutionForwardAlgorithm_v7", context);
            if (returned < 1 || perf.status != CUDNN_STATUS_SUCCESS) {
                throw InferenceException(context + ": cuDNN found no forward algorithm here");
            }
            plan->algo = perf.algo;
            CheckCudnn(cudnnGetConvolutionForwardWorkspaceSize(handle, plan->x_desc, plan->w_desc, plan->conv_desc,
                                                               plan->y_desc, plan->algo, &plan->workspace_bytes),
                       "cudnnGetConvolutionForwardWorkspaceSize", context);
            if (plan->workspace_bytes > 0) {
                CheckCuda(cudaMalloc(&plan->workspace, plan->workspace_bytes), "cudaMalloc", context);
            }
            plan->ready = true;
        }
    }

    const float alpha = 1.0F, beta = 0.0F;
    CheckCudnn(cudnnConvolutionForward(handle, &alpha, plan->x_desc, x.data, plan->w_desc, w.data, plan->conv_desc,
                                       plan->algo, plan->workspace, plan->workspace_bytes, &beta, plan->y_desc,
                                       out.data),
               "cudnnConvolutionForward", context);

    if (hasBias) {
        int64_t spatialCount = 1;
        for (std::size_t i = 0; i < spatial; ++i) {
            spatialCount *= out.dims[2 + i];
        }
        const int64_t count = n_batch * outChannels * spatialCount;
        const int block = 256;
        const int grid = static_cast<int>((count + block - 1) / block);
        BiasAddKernel<<<grid, block>>>(static_cast<float*>(out.data), bias, n_batch, outChannels, spatialCount);
        CheckCuda(cudaGetLastError(), "BiasAddKernel launch", context);
    }
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace cuda_kernels
} // namespace nert
