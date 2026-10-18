// AveragePool on OpenCL: the ONNX NCHW average pooling window.
//
// Mirrors the CPU oracle (via the verified CUDA row): kernel_shape
// required, unit strides default, dilations fixed at 1, explicit or
// automatic pads, ceil_mode output sizing. The divisor counts padded
// positions inside the padded input with count_include_pad, but never
// window positions that ceil_mode pushes past it; without it the
// divisor counts only in-bounds positions. The OpenCL C source
// compiles once per process via the shared runtime.

#include "nert/device.hpp"
#include "runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

namespace nert {
namespace opencl_kernels {
namespace {

void CheckCl(cl_int status, const char* what, const std::string& context) {
    if (status != CL_SUCCESS) {
        throw InferenceException(context + ": " + what + " failed (OpenCL status " + std::to_string(status) + ")");
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

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback, const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute '" + name + "' must be an integer");
            }
            return *value;
        }
    }
    return fallback;
}

std::vector<int64_t> IntListAttribute(const Node& node, const std::string& name, const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::vector<int64_t>* value = std::get_if<std::vector<int64_t>>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute '" + name + "' must be an int list");
            }
            return *value;
        }
    }
    return {};
}

std::string StringAttribute(const Node& node, const std::string& name, const std::string& fallback,
                            const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::string* value = std::get_if<std::string>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute '" + name + "' must be a string");
            }
            return *value;
        }
    }
    return fallback;
}

struct PoolParams {
    cl_long n_batch;
    cl_long channels;
    cl_long in_h;
    cl_long in_w;
    cl_long out_h;
    cl_long out_w;
    cl_long kh;
    cl_long kw;
    cl_long sh;
    cl_long sw;
    cl_long ph_b;
    cl_long ph_e;
    cl_long pw_b;
    cl_long pw_e;
    cl_int count_include_pad;
};

constexpr const char* kSource = R"CLC(
typedef struct {
    long n_batch; long channels; long in_h; long in_w; long out_h; long out_w;
    long kh; long kw; long sh; long sw; long ph_b; long ph_e; long pw_b; long pw_e;
    int count_include_pad;
} PoolParams;
__kernel void avgpool(__global const float* x, __global float* y, long count, PoolParams p) {
    long i = get_global_id(0);
    if (i < count) {
        long ow = i % p.out_w;
        long oh = (i / p.out_w) % p.out_h;
        long c = (i / (p.out_w * p.out_h)) % p.channels;
        long n = i / (p.out_w * p.out_h * p.channels);
        __global const float* channel = x + ((n * p.channels + c) * p.in_h) * p.in_w;
        long hStart = oh * p.sh - p.ph_b;
        long hEnd = hStart + p.kh < p.in_h + p.ph_e ? hStart + p.kh : p.in_h + p.ph_e;
        long h0 = hStart > 0 ? hStart : 0;
        long h1 = hEnd < p.in_h ? hEnd : p.in_h;
        long wStart = ow * p.sw - p.pw_b;
        long wEnd = wStart + p.kw < p.in_w + p.pw_e ? wStart + p.kw : p.in_w + p.pw_e;
        long w0 = wStart > 0 ? wStart : 0;
        long w1 = wEnd < p.in_w ? wEnd : p.in_w;
        float sum = 0.0f;
        for (long ih = h0; ih < h1; ++ih) {
            for (long iw = w0; iw < w1; ++iw) {
                sum += channel[ih * p.in_w + iw];
            }
        }
        long divisor = p.count_include_pad != 0 ? (hEnd - hStart) * (wEnd - wStart) : (h1 - h0) * (w1 - w0);
        y[i] = divisor > 0 ? sum / (float)divisor : 0.0f;
    }
}
)CLC";

void Launch(cl_kernel kernel, cl_mem x, cl_mem y, int64_t outCount, const PoolParams& p, const std::string& context) {
    cl_long count = static_cast<cl_long>(outCount);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(count), &count), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(p), &p), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(outCount);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
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

    const std::vector<int64_t> kernel = IntListAttribute(node, "kernel_shape", context);
    if (kernel.size() != 2) {
        throw InferenceException(context + ": kernel_shape must have two spatial entries");
    }
    std::vector<int64_t> strides = IntListAttribute(node, "strides", context);
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
    const bool ceilMode = IntAttribute(node, "ceil_mode", 0, context) != 0;
    const bool countIncludePad = IntAttribute(node, "count_include_pad", 0, context) != 0;

    const std::string autoPad = StringAttribute(node, "auto_pad", "NOTSET", context);
    const std::vector<int64_t> in_spatial = {inHeight, inWidth};
    const std::vector<int64_t> padsAttr = IntListAttribute(node, "pads", context);
    const std::vector<int64_t> dilations = {1, 1};
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

    auto windowOutput = [&](int64_t in, int64_t k, int64_t pb, int64_t pe, int64_t stride) {
        const int64_t effective = k; // dilations fixed at 1 per the oracle
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

    PoolParams p{};
    p.n_batch = n_batch;
    p.channels = channels;
    p.in_h = inHeight;
    p.in_w = inWidth;
    p.out_h = outHeight;
    p.out_w = outWidth;
    p.kh = kernel[0];
    p.kw = kernel[1];
    p.sh = strides[0];
    p.sw = strides[1];
    p.ph_b = pad_begin[0];
    p.ph_e = pad_end[0];
    p.pw_b = pad_begin[1];
    p.pw_e = pad_end[1];
    p.count_include_pad = countIncludePad ? 1 : 0;
    cl_mem xm = reinterpret_cast<cl_mem>(const_cast<void*>(x.data));
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    Launch(opencl_runtime::KernelFor("AveragePool", kSource, "avgpool"), xm, ym, outCount, p, context);
}

} // namespace opencl_kernels
} // namespace nert
