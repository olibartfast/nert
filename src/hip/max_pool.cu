// Hipified from src/cuda/max_pool.cu via hipify-perl, hand-reviewed.
// MaxPool on HIP: the ONNX NCHW max pooling window, one thread per output
// element.
//
// Mirrors the CPU oracle: rank-4 float32 input, required 2-entry
// kernel_shape, default unit strides/dilations, explicit or automatic pads
// (NOTSET/VALID/SAME_UPPER/SAME_LOWER with independent begin/end),
// ceil_mode output sizing, -inf for out-of-bounds samples. Only the primary
// output is written; extra outputs (indices) are ignored like the oracle.

#include "nert/device.hpp"

#include <algorithm>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <limits>
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

struct PoolParams {
    int64_t n_batch;
    int64_t channels;
    int64_t in_h;
    int64_t in_w;
    int64_t out_h;
    int64_t out_w;
    int64_t kh;
    int64_t kw;
    int64_t sh;
    int64_t sw;
    int64_t dh;
    int64_t dw;
    int64_t ph_b;
    int64_t ph_e;
    int64_t pw_b;
    int64_t pw_e;
};

__global__ void MaxPoolKernel(const float* x, float* y, int64_t count, PoolParams p, float negInf) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const int64_t ow = i % p.out_w;
        const int64_t oh = (i / p.out_w) % p.out_h;
        const int64_t c = (i / (p.out_w * p.out_h)) % p.channels;
        const int64_t n = i / (p.out_w * p.out_h * p.channels);
        const float* channel = x + ((n * p.channels + c) * p.in_h) * p.in_w;
        const int64_t rowBase = oh * p.sh - p.ph_b;
        const int64_t colBase = ow * p.sw - p.pw_b;
        float best = negInf;
        for (int64_t kh = 0; kh < p.kh; ++kh) {
            const int64_t ih = rowBase + kh * p.dh;
            if (ih < 0 || ih >= p.in_h) {
                continue;
            }
            for (int64_t kw = 0; kw < p.kw; ++kw) {
                const int64_t iw = colBase + kw * p.dw;
                if (iw < 0 || iw >= p.in_w) {
                    continue;
                }
                const float v = channel[ih * p.in_w + iw];
                if (v > best) {
                    best = v;
                }
            }
        }
        y[i] = best;
    }
}

} // namespace

void MaxPool(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "MaxPool (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.empty()) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& x = inputs[0];
    const TensorView& out = outputs[0];
    if (x.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (x.dims.size() != 4) {
        throw InferenceException(context + ": input must be rank 4 (NCHW)");
    }
    (void)ElementCount(x.dims, context);
    const int64_t outCount = ElementCount(out.dims, context);

    const int64_t n_batch = x.dims[0];
    const int64_t channels = x.dims[1];
    const int64_t inHeight = x.dims[2];
    const int64_t inWidth = x.dims[3];

    const std::vector<int64_t> kernel = IntListAttribute(node, "kernel_shape", "MaxPool");
    if (kernel.empty()) {
        throw InferenceException(context + ": kernel_shape is required");
    }
    if (kernel.size() != 2) {
        throw InferenceException(context + ": kernel_shape must have two spatial entries");
    }
    std::vector<int64_t> strides = IntListAttribute(node, "strides", "MaxPool");
    std::vector<int64_t> dilations = IntListAttribute(node, "dilations", "MaxPool");
    if (strides.empty()) {
        strides = {1, 1};
    }
    if (dilations.empty()) {
        dilations = {1, 1};
    }
    if (strides.size() != 2 || dilations.size() != 2) {
        throw InferenceException(context + ": stride and dilation must each have two spatial entries");
    }
    for (std::size_t i = 0; i < 2; ++i) {
        if (kernel[i] <= 0) {
            throw InferenceException(context + ": kernel dimensions must be positive");
        }
        if (strides[i] <= 0 || dilations[i] <= 0) {
            throw InferenceException(context + ": stride and dilation must be positive");
        }
    }
    const bool ceilMode = IntAttribute(node, "ceil_mode", 0, "MaxPool") != 0;

    // Pads: same begin/end resolution as the oracle (NOTSET/VALID/SAME_*).
    const std::string autoPad = StringAttribute(node, "auto_pad", "NOTSET", "MaxPool");
    const std::vector<int64_t> in_spatial = {inHeight, inWidth};
    const std::vector<int64_t> padsAttr = IntListAttribute(node, "pads", "MaxPool");
    std::vector<int64_t> pad_begin(2, 0), pad_end(2, 0);
    if (autoPad != "VALID") {
        if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
            for (std::size_t i = 0; i < 2; ++i) {
                const int64_t out = (in_spatial[i] + strides[i] - 1) / strides[i];
                const int64_t effective = dilations[i] * (kernel[i] - 1) + 1;
                int64_t total = (out - 1) * strides[i] + effective - in_spatial[i];
                if (total < 0) {
                    total = 0;
                }
                if (autoPad == "SAME_UPPER") {
                    pad_begin[i] = total / 2;
                    pad_end[i] = total - pad_begin[i];
                } else {
                    pad_end[i] = total / 2;
                    pad_begin[i] = total - pad_end[i];
                }
            }
        } else if (!padsAttr.empty()) {
            if (padsAttr.size() != 4) {
                throw InferenceException(context + ": pads must have two entries per spatial dimension");
            }
            pad_begin = {padsAttr[0], padsAttr[1]};
            pad_end = {padsAttr[2], padsAttr[3]};
        }
    }

    auto windowOutput = [&](int64_t in, int64_t k, int64_t dilation, int64_t pb, int64_t pe, int64_t stride) {
        const int64_t effective = dilation * (k - 1) + 1;
        const int64_t numer = in + pb + pe - effective;
        if (numer < 0) {
            throw InferenceException(context + ": window is larger than the padded input");
        }
        const int64_t o = (ceilMode ? (numer + stride - 1) / stride : numer / stride) + 1;
        if (o <= 0) {
            throw InferenceException(context + ": computes a non-positive output dimension");
        }
        return o;
    };
    const int64_t outHeight = windowOutput(inHeight, kernel[0], dilations[0], pad_begin[0], pad_end[0], strides[0]);
    const int64_t outWidth = windowOutput(inWidth, kernel[1], dilations[1], pad_begin[1], pad_end[1], strides[1]);
    if (out.dims != std::vector<int64_t>{n_batch, channels, outHeight, outWidth}) {
        throw InferenceException(context + ": output shape does not match the pooling window");
    }
    if (x.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (outCount == 0) {
        return;
    }

    PoolParams p{n_batch,      channels,   inHeight,     inWidth,    outHeight,    outWidth,
                 kernel[0],    kernel[1],  strides[0],   strides[1], dilations[0], dilations[1],
                 pad_begin[0], pad_end[0], pad_begin[1], pad_end[1]};
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    const float negInf = -std::numeric_limits<float>::infinity();
    MaxPoolKernel<<<grid, block>>>(static_cast<const float*>(x.data), static_cast<float*>(out.data), outCount, p,
                                   negInf);
    CheckHip(hipGetLastError(), "MaxPoolKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace hip_kernels
} // namespace nert
