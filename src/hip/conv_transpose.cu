// ConvTranspose on HIP (first-party, no library): the ONNX NCHW transposed
// convolution (gradient of Conv).
//
// Mirrors the CPU oracle's validation (groups, strides, dilations,
// output_padding, explicit pads). Each output element gathers its
// contributors by inverting the scatter map (oh = ih*stride - pad + kh*d),
// so one thread owns one output and no atomics are needed. Double
// accumulation rounds each output once to float32 like the oracle. No MIOpen
// dependency, so the row runs on both the NVIDIA and AMD HIP backends.

#include "nert/device.hpp"

#include <cstdint>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

namespace nert {
namespace hip_kernels {
namespace {

void CheckHip(hipError_t status, const char* what, const std::string& context) {
    if (status != hipSuccess) {
        throw InferenceException(context + ": " + what + " failed: " + hipGetErrorString(status));
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

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback, const std::string& op) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(op + ": attribute '" + name + "' must be an integer (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

std::vector<int64_t> IntListAttribute(const Node& node, const std::string& name, const std::string& op) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::vector<int64_t>* value = std::get_if<std::vector<int64_t>>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(op + ": attribute '" + name + "' must be an int list (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return {};
}

std::string StringAttribute(const Node& node, const std::string& name, const std::string& fallback,
                            const std::string& op) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::string* value = std::get_if<std::string>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(op + ": attribute '" + name + "' must be a string (node '" + node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

struct ConvTransposeParams {
    int64_t n_batch;
    int64_t in_channels;
    int64_t out_channels;
    int64_t out_per_group;
    int64_t in_per_group;
    int64_t in_h;
    int64_t in_w;
    int64_t out_h;
    int64_t out_w;
    int64_t k_h;
    int64_t k_w;
    int64_t stride_h;
    int64_t stride_w;
    int64_t dilation_h;
    int64_t dilation_w;
    int64_t pad_top;
    int64_t pad_left;
    int has_bias;
};

__global__ void ConvTransposeKernel(const float* x, const float* w, const float* bias, float* y, int64_t count,
                                    ConvTransposeParams p) {
    const int64_t inPlane = p.in_h * p.in_w;
    const int64_t outPlane = p.out_h * p.out_w;
    const int64_t kPlane = p.k_h * p.k_w;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        int64_t rem = i;
        const int64_t ow = rem % p.out_w;
        rem /= p.out_w;
        const int64_t oh = rem % p.out_h;
        rem /= p.out_h;
        const int64_t m = rem % p.out_channels;
        const int64_t n = rem / p.out_channels;
        const int64_t g = m / p.out_per_group;
        const int64_t oc = m % p.out_per_group;
        double acc = p.has_bias != 0 ? static_cast<double>(bias[m]) : 0.0;
        for (int64_t ic = 0; ic < p.in_per_group; ++ic) {
            const int64_t c = g * p.in_per_group + ic;
            const float* xPlane = x + (n * p.in_channels + c) * inPlane;
            const float* wBase = w + (c * p.out_per_group + oc) * kPlane;
            for (int64_t kh = 0; kh < p.k_h; ++kh) {
                // oh = ih*stride - pad + kh*dilation, inverted for the input row.
                const int64_t numerH = oh + p.pad_top - kh * p.dilation_h;
                if (numerH < 0 || numerH % p.stride_h != 0) {
                    continue;
                }
                const int64_t ih = numerH / p.stride_h;
                if (ih >= p.in_h) {
                    continue;
                }
                for (int64_t kw = 0; kw < p.k_w; ++kw) {
                    const int64_t numerW = ow + p.pad_left - kw * p.dilation_w;
                    if (numerW < 0 || numerW % p.stride_w != 0) {
                        continue;
                    }
                    const int64_t iw = numerW / p.stride_w;
                    if (iw >= p.in_w) {
                        continue;
                    }
                    acc += static_cast<double>(xPlane[ih * p.in_w + iw]) * static_cast<double>(wBase[kh * p.k_w + kw]);
                }
            }
        }
        y[(n * p.out_channels + m) * outPlane + oh * p.out_w + ow] = static_cast<float>(acc);
    }
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

    const int64_t group = IntAttribute(node, "group", 1, "ConvTranspose");
    const int64_t nBatch = x.dims[0];
    const int64_t inChannels = x.dims[1];
    const int64_t inHeight = x.dims[2];
    const int64_t inWidth = x.dims[3];
    if (group <= 0 || inChannels % group != 0 || w.dims[0] != inChannels) {
        throw InferenceException(context + ": weight channels do not match the input and group");
    }
    const int64_t outPerGroup = w.dims[1];
    const int64_t outChannels = outPerGroup * group;
    const int64_t inPerGroup = inChannels / group;
    const int64_t kH = w.dims[2];
    const int64_t kW = w.dims[3];

    std::vector<int64_t> strides = IntListAttribute(node, "strides", "ConvTranspose");
    std::vector<int64_t> dilations = IntListAttribute(node, "dilations", "ConvTranspose");
    std::vector<int64_t> outputPadding = IntListAttribute(node, "output_padding", "ConvTranspose");
    std::vector<int64_t> pads = IntListAttribute(node, "pads", "ConvTranspose");
    if (strides.empty()) {
        strides = {1, 1};
    }
    if (dilations.empty()) {
        dilations = {1, 1};
    }
    if (outputPadding.empty()) {
        outputPadding = {0, 0};
    }
    if (pads.empty() || StringAttribute(node, "auto_pad", "NOTSET", "ConvTranspose") == "VALID") {
        pads = {0, 0, 0, 0};
    }
    if (strides.size() != 2 || dilations.size() != 2 || outputPadding.size() != 2 || pads.size() != 4) {
        throw InferenceException(context + ": strides, dilations, output_padding, and pads must be 2-D");
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

    const float* bData = bias != nullptr ? static_cast<const float*>(bias->data) : nullptr;
    ConvTransposeParams p{};
    p.n_batch = nBatch;
    p.in_channels = inChannels;
    p.out_channels = outChannels;
    p.out_per_group = outPerGroup;
    p.in_per_group = inPerGroup;
    p.in_h = inHeight;
    p.in_w = inWidth;
    p.out_h = outHeight;
    p.out_w = outWidth;
    p.k_h = kH;
    p.k_w = kW;
    p.stride_h = strides[0];
    p.stride_w = strides[1];
    p.dilation_h = dilations[0];
    p.dilation_w = dilations[1];
    p.pad_top = pads[0];
    p.pad_left = pads[1];
    p.has_bias = bias != nullptr ? 1 : 0;
    const int64_t outCount = nBatch * outChannels * outHeight * outWidth;

    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    ConvTransposeKernel<<<grid, block>>>(static_cast<const float*>(x.data), static_cast<const float*>(w.data), bData,
                                         static_cast<float*>(out.data), outCount, p);
    CheckHip(hipGetLastError(), "ConvTransposeKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace hip_kernels
} // namespace nert
