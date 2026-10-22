// Hipified from src/cuda/average_pool.cu via hipify-perl, hand-reviewed.
// AveragePool on HIP: the ONNX NCHW average pooling window, one thread per
// output element.
//
// Mirrors the CPU oracle: rank-4 float32 input, required 2-entry
// kernel_shape, default unit strides, explicit or automatic pads, ceil_mode
// output sizing. The divisor counts padded positions inside the padded input
// with count_include_pad (but never window positions ceil_mode pushes past
// it); otherwise it counts in-bounds elements only. The float accumulation
// order matches the oracle's row-major sum.

#include "nert/device.hpp"

#include <algorithm>
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
    int64_t ph_b;
    int64_t ph_e;
    int64_t pw_b;
    int64_t pw_e;
    int count_include_pad;
};

__global__ void AveragePoolKernel(const float* x, float* y, int64_t count, PoolParams p) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const int64_t ow = i % p.out_w;
        const int64_t oh = (i / p.out_w) % p.out_h;
        const int64_t c = (i / (p.out_w * p.out_h)) % p.channels;
        const int64_t n = i / (p.out_w * p.out_h * p.channels);
        const float* channel = x + ((n * p.channels + c) * p.in_h) * p.in_w;
        const int64_t hStart = oh * p.sh - p.ph_b;
        const int64_t hEnd = hStart + p.kh < p.in_h + p.ph_e ? hStart + p.kh : p.in_h + p.ph_e;
        const int64_t h0 = hStart > 0 ? hStart : 0;
        const int64_t h1 = hEnd < p.in_h ? hEnd : p.in_h;
        const int64_t wStart = ow * p.sw - p.pw_b;
        const int64_t wEnd = wStart + p.kw < p.in_w + p.pw_e ? wStart + p.kw : p.in_w + p.pw_e;
        const int64_t w0 = wStart > 0 ? wStart : 0;
        const int64_t w1 = wEnd < p.in_w ? wEnd : p.in_w;
        float sum = 0.0F;
        for (int64_t ih = h0; ih < h1; ++ih) {
            for (int64_t iw = w0; iw < w1; ++iw) {
                sum += channel[ih * p.in_w + iw];
            }
        }
        const int64_t divisor = p.count_include_pad != 0 ? (hEnd - hStart) * (wEnd - wStart) : (h1 - h0) * (w1 - w0);
        y[i] = divisor > 0 ? sum / static_cast<float>(divisor) : 0.0F;
    }
}

} // namespace

void AveragePool(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "AveragePool (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
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

    const std::vector<int64_t> kernel = IntListAttribute(node, "kernel_shape", "AveragePool");
    if (kernel.size() != 2) {
        throw InferenceException(context + ": kernel_shape must have two spatial entries");
    }
    std::vector<int64_t> strides = IntListAttribute(node, "strides", "AveragePool");
    if (strides.empty()) {
        strides = {1, 1};
    }
    if (strides.size() != 2) {
        throw InferenceException(context + ": strides must have two spatial entries");
    }
    for (std::size_t i = 0; i < 2; ++i) {
        if (kernel[i] <= 0 || strides[i] <= 0) {
            throw InferenceException(context + ": kernel and stride must be positive");
        }
    }
    const bool ceilMode = IntAttribute(node, "ceil_mode", 0, "AveragePool") != 0;
    const bool countIncludePad = IntAttribute(node, "count_include_pad", 0, "AveragePool") != 0;

    // Pads: same begin/end resolution as the oracle (NOTSET/VALID/SAME_*).
    const std::string autoPad = StringAttribute(node, "auto_pad", "NOTSET", "AveragePool");
    const std::vector<int64_t> padsAttr = IntListAttribute(node, "pads", "AveragePool");
    std::vector<int64_t> pad_begin(2, 0), pad_end(2, 0);
    if (autoPad != "VALID") {
        if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
            for (std::size_t i = 0; i < 2; ++i) {
                const int64_t in = i == 0 ? inHeight : inWidth;
                const int64_t o = (in + strides[i] - 1) / strides[i];
                const int64_t effective = kernel[i]; // no dilations on this path
                int64_t total = (o - 1) * strides[i] + effective - in;
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

    auto windowOutput = [&](int64_t in, int64_t k, int64_t pb, int64_t pe, int64_t stride) {
        const int64_t numer = in + pb + pe - k;
        if (numer < 0) {
            throw InferenceException(context + ": window is larger than the padded input");
        }
        const int64_t o = (ceilMode ? (numer + stride - 1) / stride : numer / stride) + 1;
        if (o <= 0) {
            throw InferenceException(context + ": computes a non-positive output dimension");
        }
        return o;
    };
    const int64_t outHeight = windowOutput(inHeight, kernel[0], pad_begin[0], pad_end[0], strides[0]);
    const int64_t outWidth = windowOutput(inWidth, kernel[1], pad_begin[1], pad_end[1], strides[1]);
    if (out.dims != std::vector<int64_t>{n_batch, channels, outHeight, outWidth}) {
        throw InferenceException(context + ": output shape does not match the pooling window");
    }
    if (x.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (outCount == 0) {
        return;
    }

    PoolParams p{n_batch,      channels,   inHeight,     inWidth,    outHeight,
                 outWidth,     kernel[0],  kernel[1],    strides[0], strides[1],
                 pad_begin[0], pad_end[0], pad_begin[1], pad_end[1], countIncludePad ? 1 : 0};
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    AveragePoolKernel<<<grid, block>>>(static_cast<const float*>(x.data), static_cast<float*>(out.data), outCount, p);
    CheckHip(hipGetLastError(), "AveragePoolKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace hip_kernels
} // namespace nert
