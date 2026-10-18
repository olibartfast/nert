// Conv on OpenCL: the ONNX cross-correlation over one, two, or three
// spatial axes, with groups, strides, dilations and explicit or automatic
// padding.
//
// Mirrors the CPU oracle exactly: input [N,C,spatial...], weight
// [M, C/group, k...], optional bias [M]; output
// floor((in + pad_begin + pad_end - dilation*(k-1) - 1)/stride) + 1.
// Rank 3/4/5 lift to three spatial axes with unit leading axes. One
// work-item per output element, float accumulation with one rounding.

#include "nert/device.hpp"
#include "runtime.hpp"

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

int64_t WindowOutput(const std::string& context, int64_t in, int64_t kernel, int64_t dilation, int64_t pad_begin,
                     int64_t pad_end, int64_t stride) {
    const int64_t effective = dilation * (kernel - 1) + 1;
    const int64_t numer = in + pad_begin + pad_end - effective;
    if (numer < 0) {
        throw InferenceException(context + ": window is larger than the padded input");
    }
    const int64_t out = numer / stride + 1;
    if (out <= 0) {
        throw InferenceException(context + ": computes a non-positive output dimension");
    }
    return out;
}

struct ConvParams {
    cl_long n_batch;
    cl_long in_channels;
    cl_long out_channels;
    cl_long channels_per_group;
    cl_long outputs_per_group;
    cl_long in_d;
    cl_long in_h;
    cl_long in_w;
    cl_long kd;
    cl_long kh;
    cl_long kw;
    cl_long od;
    cl_long oh;
    cl_long ow;
    cl_long sd;
    cl_long sh;
    cl_long sw;
    cl_long dd;
    cl_long dh;
    cl_long dw;
    cl_long pd;
    cl_long ph;
    cl_long pw;
    cl_int has_bias;
};

constexpr const char* kSource = R"CLC(
typedef struct {
    long n_batch; long in_channels; long out_channels;
    long channels_per_group; long outputs_per_group;
    long in_d; long in_h; long in_w;
    long kd; long kh; long kw;
    long od; long oh; long ow;
    long sd; long sh; long sw;
    long dd; long dh; long dw;
    long pd; long ph; long pw;
    int has_bias;
} ConvParams;
__kernel void conv_f(__global const float* x, __global const float* w, __global const float* b,
                     __global float* y, ConvParams p) {
    long total = p.n_batch * p.out_channels * p.od * p.oh * p.ow;
    long idx = get_global_id(0);
    if (idx >= total) {
        return;
    }
    long ow = idx % p.ow;
    long oh = (idx / p.ow) % p.oh;
    long od = (idx / (p.ow * p.oh)) % p.od;
    long m = (idx / (p.ow * p.oh * p.od)) % p.out_channels;
    long n = idx / (p.ow * p.oh * p.od * p.out_channels);

    float acc = p.has_bias != 0 ? b[m] : 0.0f;
    long channelBase = (m / p.outputs_per_group) * p.channels_per_group;
    long inPlane = p.in_d * p.in_h * p.in_w;
    long kPlane = p.kd * p.kh * p.kw;
    for (long kc = 0; kc < p.channels_per_group; ++kc) {
        __global const float* xc = x + (n * p.in_channels + channelBase + kc) * inPlane;
        __global const float* wc = w + (m * p.channels_per_group + kc) * kPlane;
        for (long kd = 0; kd < p.kd; ++kd) {
            long id = od * p.sd - p.pd + kd * p.dd;
            if (id < 0 || id >= p.in_d) {
                continue;
            }
            for (long kh = 0; kh < p.kh; ++kh) {
                long ih = oh * p.sh - p.ph + kh * p.dh;
                if (ih < 0 || ih >= p.in_h) {
                    continue;
                }
                __global const float* xrow = xc + (id * p.in_h + ih) * p.in_w;
                __global const float* wrow = wc + (kd * p.kh + kh) * p.kw;
                for (long kw = 0; kw < p.kw; ++kw) {
                    long iw = ow * p.sw - p.pw + kw * p.dw;
                    if (iw < 0 || iw >= p.in_w) {
                        continue;
                    }
                    acc += xrow[iw] * wrow[kw];
                }
            }
        }
    }
    y[idx] = acc;
}
)CLC";

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
    const int64_t channelsPerGroup = inChannels / group;
    const int64_t outputsPerGroup = outChannels / group;

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

    std::vector<int64_t> expected = {n_batch, outChannels};
    for (std::size_t i = 0; i < spatial; ++i) {
        expected.push_back(
            WindowOutput(context, in_spatial[i], kernel[i], dilations[i], pad_begin[i], pad_end[i], strides[i]));
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the convolution");
    }

    const TensorView* bias = nullptr;
    if (hasBias) {
        const TensorView& biasView = inputs[2];
        if (biasView.dims.size() != 1 || biasView.dims[0] != outChannels) {
            throw InferenceException(context + ": bias must be rank-1 of length M");
        }
        bias = &biasView;
    }
    if (x.data == nullptr || w.data == nullptr || out.data == nullptr || (hasBias && bias->data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    const std::size_t lift = 3 - spatial;
    auto lifted = [lift](const std::vector<int64_t>& v, int64_t fill) {
        std::vector<int64_t> r(lift, fill);
        r.insert(r.end(), v.begin(), v.end());
        return r;
    };
    const std::vector<int64_t> in3 = lifted(in_spatial, 1);
    const std::vector<int64_t> k3 = lifted(kernel, 1);
    const std::vector<int64_t> o3 = lifted(std::vector<int64_t>(expected.begin() + 2, expected.end()), 1);
    const std::vector<int64_t> s3 = lifted(strides, 1);
    const std::vector<int64_t> d3 = lifted(dilations, 1);
    const std::vector<int64_t> p3 = lifted(pad_begin, 0);

    ConvParams p{};
    p.n_batch = n_batch;
    p.in_channels = inChannels;
    p.out_channels = outChannels;
    p.channels_per_group = channelsPerGroup;
    p.outputs_per_group = outputsPerGroup;
    p.in_d = in3[0];
    p.in_h = in3[1];
    p.in_w = in3[2];
    p.kd = k3[0];
    p.kh = k3[1];
    p.kw = k3[2];
    p.od = o3[0];
    p.oh = o3[1];
    p.ow = o3[2];
    p.sd = s3[0];
    p.sh = s3[1];
    p.sw = s3[2];
    p.dd = d3[0];
    p.dh = d3[1];
    p.dw = d3[2];
    p.pd = p3[0];
    p.ph = p3[1];
    p.pw = p3[2];
    p.has_bias = hasBias ? 1 : 0;

    cl_mem xm = reinterpret_cast<cl_mem>(const_cast<void*>(x.data));
    cl_mem wm = reinterpret_cast<cl_mem>(const_cast<void*>(w.data));
    cl_mem bm = hasBias ? reinterpret_cast<cl_mem>(const_cast<void*>(bias->data)) : nullptr;
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    cl_kernel kernelHandle = opencl_runtime::KernelFor("Conv", kSource, "conv_f");
    CheckCl(clSetKernelArg(kernelHandle, 0, sizeof(xm), &xm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernelHandle, 1, sizeof(wm), &wm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernelHandle, 2, sizeof(bm), &bm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernelHandle, 3, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernelHandle, 4, sizeof(p), &p), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(n_batch * outChannels * o3[0] * o3[1] * o3[2]);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernelHandle, 1, nullptr, &global, nullptr, 0, nullptr,
                                   nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
