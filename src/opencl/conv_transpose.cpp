// ConvTranspose on OpenCL: the ONNX NCHW transposed convolution (gradient of
// Conv) as a direct (gather) convolution.
//
// Mirrors the CPU oracle exactly: input [N,C,H,W], weight [C, M/group, kH, kW],
// optional rank-1 bias [M]; output spatial size is stride*(in-1) +
// output_padding + dilation*(k-1) + 1 - pad_begin - pad_end with explicit
// padding (auto_pad NOTSET or VALID). output_shape and kernel_shape are
// accepted but not applied, like the oracle: the output size is fixed by the
// formula and the output view must match it. Groups are supported in general
// (the surveyed export uses group=1).
//
// One work-item per output element. For each kernel tap the source coordinate
// is recovered from the output coordinate (a tap lands on this element exactly
// when the stride divides the offset and the source stays inside the input),
// products accumulate in double with one float32 rounding, and the bias is
// added last. Weights and bias stay in their device buffers and are indexed
// directly, so no host readback is needed. The fp64 guard follows
// global_average_pool: the device must report CL_DEVICE_DOUBLE_FP_CONFIG
// support before the kernel is built and launched.

#include "nert/device.hpp"
#include "runtime.hpp"

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

// Reads an int64 attribute, or returns the fallback when the attribute is absent.
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

// Reads an int list attribute, or an empty list when the attribute is absent.
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

// Reads a string attribute, or returns the fallback when the attribute is absent.
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

struct ConvTransposeParams {
    cl_long n_batch;
    cl_long in_channels;
    cl_long out_channels;
    cl_long in_per_group;
    cl_long out_per_group;
    cl_long in_h;
    cl_long in_w;
    cl_long out_h;
    cl_long out_w;
    cl_long kh;
    cl_long kw;
    cl_long sh;
    cl_long sw;
    cl_long dh;
    cl_long dw;
    cl_long ph;
    cl_long pw;
    cl_int has_bias;
};

constexpr const char* kSource = R"CLC(
typedef struct {
    long n_batch; long in_channels; long out_channels;
    long in_per_group; long out_per_group;
    long in_h; long in_w; long out_h; long out_w;
    long kh; long kw;
    long sh; long sw; long dh; long dw;
    long ph; long pw;
    int has_bias;
} ConvTransposeParams;
// Gather transposed convolution: output element (n, m, oh, ow) sums every
// input tap that maps onto it. Tap (kh, kw) of input row ih contributes to
// this element exactly when oh + ph - kh*dh == ih*sh (the stride divides the
// offset) with ih inside the input, and likewise in width.
__kernel void conv_transpose_f(__global const float* x, __global const float* w, __global const float* b,
                               __global float* y, ConvTransposeParams p) {
    long total = p.n_batch * p.out_channels * p.out_h * p.out_w;
    long idx = get_global_id(0);
    if (idx >= total) {
        return;
    }
    long ow = idx % p.out_w;
    long oh = (idx / p.out_w) % p.out_h;
    long m = (idx / (p.out_w * p.out_h)) % p.out_channels;
    long n = idx / (p.out_w * p.out_h * p.out_channels);

    long g = m / p.out_per_group;
    long oc = m % p.out_per_group;
    long inPlane = p.in_h * p.in_w;
    long kPlane = p.kh * p.kw;
    double acc = 0.0;
    for (long ic = 0; ic < p.in_per_group; ++ic) {
        long c = g * p.in_per_group + ic;
        __global const float* xc = x + (n * p.in_channels + c) * inPlane;
        __global const float* wc = w + (c * p.out_per_group + oc) * kPlane;
        for (long kh = 0; kh < p.kh; ++kh) {
            long t = oh + p.ph - kh * p.dh;
            if (t < 0 || t % p.sh != 0) {
                continue;
            }
            long ih = t / p.sh;
            if (ih >= p.in_h) {
                continue;
            }
            for (long kw = 0; kw < p.kw; ++kw) {
                long u = ow + p.pw - kw * p.dw;
                if (u < 0 || u % p.sw != 0) {
                    continue;
                }
                long iw = u / p.sw;
                if (iw >= p.in_w) {
                    continue;
                }
                acc += (double)xc[ih * p.in_w + iw] * (double)wc[kh * p.kw + kw];
            }
        }
    }
    if (p.has_bias != 0) {
        acc += (double)b[m];
    }
    y[idx] = (float)acc;
}
)CLC";

} // namespace

void ConvTranspose(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ConvTranspose (node '" + node.name + "')";
    if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected input, weight, optional bias, and one output");
    }
    const TensorView& x = inputs[0];
    const TensorView& w = inputs[1];
    // A third input without a buffer counts as an absent bias, like the oracle.
    const TensorView* bias = inputs.size() == 3 && inputs[2].data != nullptr ? &inputs[2] : nullptr;
    const TensorView& out = outputs[0];
    if (x.dtype != DataType::Float32 || w.dtype != DataType::Float32 || out.dtype != DataType::Float32 ||
        (bias != nullptr && bias->dtype != DataType::Float32)) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (x.dims.size() != 4 || w.dims.size() != 4) {
        // Rank 4 on both operands also keeps the pilot rank cap of 16 satisfied.
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
    const int64_t inPerGroup = inChannels / group;
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
    // The gather kernel recovers the source coordinate by dividing by the
    // stride, so non-positive strides cannot be supported (the scatter oracle
    // happens to accept them; every real export uses >= 1).
    if (strides[0] <= 0 || strides[1] <= 0) {
        throw InferenceException(context + ": strides must be positive");
    }

    const int64_t total = nBatch * outChannels * outHeight * outWidth;
    if (total == 0) {
        return;
    }
    // The kernel accumulates in double; OpenCL fp64 is optional, so fail with
    // the device capability named instead of a build error.
    cl_device_id device = nullptr;
    CheckCl(clGetContextInfo(opencl_runtime::Context(), CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr),
            "clGetContextInfo", context);
    cl_device_fp_config fp64 = 0;
    CheckCl(clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp64), &fp64, nullptr), "clGetDeviceInfo",
            context);
    if (fp64 == 0) {
        throw InferenceException(context + ": device has no double-precision support (cl_khr_fp64 is required)");
    }

    cl_mem xm = reinterpret_cast<cl_mem>(const_cast<void*>(x.data));
    cl_mem wm = reinterpret_cast<cl_mem>(const_cast<void*>(w.data));
    cl_mem bm = bias != nullptr ? reinterpret_cast<cl_mem>(const_cast<void*>(bias->data)) : nullptr;
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    ConvTransposeParams p{};
    p.n_batch = nBatch;
    p.in_channels = inChannels;
    p.out_channels = outChannels;
    p.in_per_group = inPerGroup;
    p.out_per_group = outPerGroup;
    p.in_h = inHeight;
    p.in_w = inWidth;
    p.out_h = outHeight;
    p.out_w = outWidth;
    p.kh = kH;
    p.kw = kW;
    p.sh = strides[0];
    p.sw = strides[1];
    p.dh = dilations[0];
    p.dw = dilations[1];
    p.ph = pads[0];
    p.pw = pads[1];
    p.has_bias = bias != nullptr ? 1 : 0;

    cl_kernel kernelHandle = opencl_runtime::KernelFor("ConvTranspose", kSource, "conv_transpose_f");
    CheckCl(clSetKernelArg(kernelHandle, 0, sizeof(xm), &xm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernelHandle, 1, sizeof(wm), &wm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernelHandle, 2, sizeof(bm), &bm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernelHandle, 3, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernelHandle, 4, sizeof(p), &p), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(total);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernelHandle, 1, nullptr, &global, nullptr, 0, nullptr,
                                   nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
