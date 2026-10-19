// Resize on OpenCL: nearest or linear resampling with the ONNX coordinate
// transforms.
//
// Mirrors the CPU oracle (src/kernels/resize.cpp): the output dims come from
// a sizes input, or floor(in * scale) from a scales input (never both), and
// each output coordinate maps through coordinate_transformation_mode
// (half_pixel, pytorch_half_pixel, align_corners, asymmetric) with nearest
// rounding per nearest_mode (round_prefer_floor, round_prefer_ceil, floor,
// ceil) or per-axis linear blending between the two neighbours. Antialiasing
// and cubic mode are rejected at load; exclude_outside and antialias are
// named and rejected if they reach this row. Data is float32, or int64 for
// nearest.
//
// The roi, scales, and sizes inputs live in device memory on this seam: the
// scales and sizes values are read back with a blocking clEnqueueReadBuffer
// before validation and never dereferenced as host pointers (roi is unused,
// as in the oracle). The per-axis source coordinates and weights are
// recomputed inside the kernel with the same double-precision formulas as
// the oracle, so the device must report double-precision support
// (cl_khr_fp64). One work-item per output element. The OpenCL C source
// compiles once per process via the shared runtime.

#include "nert/device.hpp"
#include "runtime.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

namespace nert {
namespace opencl_kernels {
namespace {

constexpr std::size_t kMaxRank = 16;

void CheckCl(cl_int status, const char* what, const std::string& context) {
    if (status != CL_SUCCESS) {
        throw InferenceException(context + ": " + what + " failed (OpenCL status " + std::to_string(status) + ")");
    }
}

int64_t ElementCount(const std::vector<int64_t>& dims, const std::string& context) {
    int64_t count = 1;
    for (int64_t dim : dims) {
        if (dim < 0) {
            throw InferenceException(context + ": negative dimension");
        }
        count *= dim;
    }
    return count;
}

std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

std::string StringAttribute(const Node& node, const std::string& name, const std::string& fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::string* value = std::get_if<std::string>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("Resize: attribute '" + name + "' must be a string (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

// Axis ids passed to the kernel, with the oracle's rejection messages.
cl_long TransformId(const std::string& transform, const std::string& context) {
    if (transform == "half_pixel") {
        return 0;
    }
    if (transform == "pytorch_half_pixel") {
        return 1;
    }
    if (transform == "align_corners") {
        return 2;
    }
    if (transform == "asymmetric") {
        return 3;
    }
    throw InferenceException(context + ": unsupported coordinate_transformation_mode '" + transform + "'");
}

cl_long NearestId(const std::string& nearest, const std::string& context) {
    if (nearest == "round_prefer_floor") {
        return 0;
    }
    if (nearest == "round_prefer_ceil") {
        return 1;
    }
    if (nearest == "floor") {
        return 2;
    }
    if (nearest == "ceil") {
        return 3;
    }
    throw InferenceException(context + ": unsupported nearest_mode '" + nearest + "'");
}

// Blocking read-back of a small device buffer (the scales and sizes values
// live on the device on this seam).
void ReadDevice(const void* device_ptr, void* host_dst, std::size_t bytes, const std::string& context) {
    cl_mem mem = reinterpret_cast<cl_mem>(const_cast<void*>(device_ptr));
    CheckCl(clEnqueueReadBuffer(opencl_runtime::Queue(), mem, CL_TRUE, 0, bytes, host_dst, 0, nullptr, nullptr),
            "clEnqueueReadBuffer", context);
}

// Per-axis launch parameters; the layout must mirror ResizeParams in the
// kernel source below.
struct ResizeParams {
    cl_long in_strides[kMaxRank];
    cl_long out_strides[kMaxRank];
    cl_long in_len[kMaxRank];
    cl_long out_len[kMaxRank];
    cl_double scale[kMaxRank];
    cl_long rank;
    cl_long linear;
    cl_long transform;
    cl_long nearest;
};

constexpr const char* kSource = R"CLC(
// Per-axis launch parameters; the layout must mirror the host ResizeParams.
typedef struct {
    long in_strides[16];
    long out_strides[16];
    long in_len[16];
    long out_len[16];
    double scale[16];
    long rank;
    long linear;
    long transform;  // 0 half_pixel, 1 pytorch_half_pixel, 2 align_corners, 3 asymmetric
    long nearest;    // 0 round_prefer_floor, 1 round_prefer_ceil, 2 floor, 3 ceil
} ResizeParams;

// The input coordinate output coordinate o samples on axis j, per the ONNX
// transform (the oracle's double-precision formulas).
double SourceCoord(const ResizeParams* p, long j, long o) {
    const double x = (double)o;
    const long out_len = p->out_len[j];
    if (p->transform == 0) {
        return (x + 0.5) / p->scale[j] - 0.5;                                       // half_pixel
    }
    if (p->transform == 1) {
        return out_len > 1 ? (x + 0.5) / p->scale[j] - 0.5 : 0.0;                   // pytorch_half_pixel
    }
    if (p->transform == 2) {
        return out_len == 1 ? 0.0 : x * (double)(p->in_len[j] - 1) / (double)(out_len - 1);  // align_corners
    }
    return x / p->scale[j];                                                         // asymmetric
}

// The clamped input index for nearest sampling (the oracle's rounding).
long NearestIdx(const ResizeParams* p, double x, long j) {
    long index = 0;
    if (p->nearest == 0) {
        const double f = floor(x);
        index = (x - f == 0.5) ? (long)f : (long)round(x);   // round_prefer_floor
    } else if (p->nearest == 1) {
        index = (long)round(x);                              // round_prefer_ceil
    } else if (p->nearest == 2) {
        index = (long)floor(x);                              // floor
    } else {
        index = (long)ceil(x);                               // ceil
    }
    index = index < 0 ? 0 : index;
    const long limit = p->in_len[j] - 1;
    return index > limit ? limit : index;
}

// One work-item per output element. Nearest copies one input value; linear
// blends the 2^rank corners with the per-axis weight of the upper neighbour,
// keeping zero-weight axes on their lower neighbour like the oracle's tap
// expansion.
__kernel void resize(__global const float* xf, __global float* yf, __global const long* xl, __global long* yl,
                     long use_int64, long out_count, ResizeParams p) {
    const long i = (long)get_global_id(0);
    if (i >= out_count) {
        return;
    }
    long lower[16];
    long upper[16];
    double weight[16];
    long remaining = i;
    long offset = 0;
    for (long j = 0; j < p.rank; ++j) {
        const long c = remaining / p.out_strides[j];
        remaining -= c * p.out_strides[j];
        const double x = SourceCoord(&p, j, c);
        if (!p.linear) {
            lower[j] = NearestIdx(&p, x, j);
            upper[j] = lower[j];
            weight[j] = 0.0;
        } else {
            const double limit = (double)(p.in_len[j] - 1);
            const double clamped = x < 0.0 ? 0.0 : (x > limit ? limit : x);
            const long lo = (long)floor(clamped);
            const long up = lo + 1 < p.in_len[j] ? lo + 1 : p.in_len[j] - 1;
            lower[j] = lo;
            upper[j] = up;
            weight[j] = up == lo ? 0.0 : clamped - (double)lo;
        }
        offset += lower[j] * p.in_strides[j];
    }
    if (use_int64 != 0) {
        yl[i] = xl[offset];
        return;
    }
    if (!p.linear) {
        yf[i] = xf[offset];
        return;
    }
    double value = 0.0;
    const long taps = (long)1 << p.rank;
    for (long mask = 0; mask < taps; ++mask) {
        double w = 1.0;
        long tap = offset;
        for (long j = 0; j < p.rank; ++j) {
            if ((mask >> j) & 1L) {
                w *= weight[j];
                tap += (upper[j] - lower[j]) * p.in_strides[j];
            } else {
                w *= 1.0 - weight[j];
            }
        }
        if (w != 0.0) {
            value += w * (double)xf[tap];
        }
    }
    yf[i] = (float)value;
}
)CLC";

} // namespace

void Resize(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Resize (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 4 || outputs.size() != 1) {
        throw InferenceException(context + ": expected an input, roi, scales, and an optional sizes input");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "antialias" || attr.name == "exclude_outside") {
            throw InferenceException(context + ": attribute '" + attr.name + "' is not supported here");
        }
    }
    const std::string mode = StringAttribute(node, "mode", "nearest");
    if (mode != "nearest" && mode != "linear") {
        throw InferenceException(context + ": Resize 'mode' must be 'nearest' or 'linear' here");
    }
    const bool linear = mode == "linear";
    if (in.dtype != DataType::Float32 && !(in.dtype == DataType::Int64 && !linear)) {
        throw InferenceException(context + ": tensors must be float32 (or int64 for nearest)");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    const std::string transform = StringAttribute(node, "coordinate_transformation_mode", "half_pixel");
    const std::string nearest = StringAttribute(node, "nearest_mode", "round_prefer_floor");
    const cl_long transform_id = TransformId(transform, context);
    const cl_long nearest_id = NearestId(nearest, context);

    const std::size_t rank = in.dims.size();
    if (rank < 1 || rank > kMaxRank) {
        throw InferenceException(context + ": Resize supports rank 1 through 16");
    }
    for (int64_t dim : in.dims) {
        if (dim <= 0) {
            throw InferenceException(context + ": non-positive dimension");
        }
    }
    const int64_t rank_i = static_cast<int64_t>(rank);

    std::vector<double> scale_vec(rank, 1.0);
    const bool has_scales =
        inputs.size() > 2 && inputs[2].dtype != DataType::Unknown && ElementCount(inputs[2].dims, context) > 0;
    const bool has_sizes = inputs.size() > 3 && inputs[3].dtype != DataType::Unknown;
    if (has_sizes && has_scales) {
        throw InferenceException(context + ": Resize takes exactly one of scales and sizes");
    }
    if (has_sizes) {
        // A sizes input fixes the output dims; the per-axis scale used by the
        // coordinate transform is then out / in, as ONNX defines it. The
        // values live on the device: read them back before validating.
        const TensorView& sizes_view = inputs[3];
        if (sizes_view.dtype != DataType::Int64 || ElementCount(sizes_view.dims, context) != rank_i ||
            sizes_view.data == nullptr) {
            throw InferenceException(context + ": Resize sizes must be int64 with one entry per input axis");
        }
        std::vector<int64_t> sizes(rank);
        ReadDevice(sizes_view.data, sizes.data(), rank * sizeof(int64_t), context);
        for (std::size_t i = 0; i < rank; ++i) {
            if (out.dims.size() != rank || out.dims[i] != sizes[i]) {
                throw InferenceException(context + ": output shape does not match the sizes input");
            }
            scale_vec[i] = static_cast<double>(static_cast<float>(sizes[i]) / static_cast<float>(in.dims[i]));
        }
    } else {
        if (!has_scales || inputs[2].dtype != DataType::Float32) {
            throw InferenceException(context + ": Resize requires a scales or sizes input");
        }
        const TensorView& scales_view = inputs[2];
        if (ElementCount(scales_view.dims, context) != rank_i) {
            throw InferenceException(context + ": Resize scales rank must match the input rank");
        }
        if (scales_view.data == nullptr) {
            throw InferenceException(context + ": null scales buffer");
        }
        std::vector<float> scales(rank);
        ReadDevice(scales_view.data, scales.data(), rank * sizeof(float), context);
        std::vector<int64_t> expected(rank, 0);
        for (std::size_t i = 0; i < rank; ++i) {
            const int64_t dim =
                static_cast<int64_t>(std::floor(static_cast<double>(in.dims[i]) * static_cast<double>(scales[i])));
            if (dim <= 0) {
                throw InferenceException(context + ": Resize produces a non-positive output dim");
            }
            expected[i] = dim;
            scale_vec[i] = static_cast<double>(scales[i]);
        }
        if (out.dims != expected) {
            throw InferenceException(context + ": output shape does not match the scaled input");
        }
    }

    const int64_t out_count = ElementCount(out.dims, context);
    if (out_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    if (out_count == 0) {
        return;
    }

    // The kernel computes the source coordinates and weights in double;
    // OpenCL fp64 is optional, so fail with the device capability named
    // instead of a build error.
    cl_device_id device = nullptr;
    CheckCl(clGetContextInfo(opencl_runtime::Context(), CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr),
            "clGetContextInfo", context);
    cl_device_fp_config fp64 = 0;
    CheckCl(clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp64), &fp64, nullptr), "clGetDeviceInfo",
            context);
    if (fp64 == 0) {
        throw InferenceException(context + ": device has no double-precision support (cl_khr_fp64 is required)");
    }

    ResizeParams params{};
    const std::vector<int64_t> in_strides = Strides(in.dims);
    const std::vector<int64_t> out_strides = Strides(out.dims);
    for (std::size_t i = 0; i < rank; ++i) {
        params.in_strides[i] = static_cast<cl_long>(in_strides[i]);
        params.out_strides[i] = static_cast<cl_long>(out_strides[i]);
        params.in_len[i] = static_cast<cl_long>(in.dims[i]);
        params.out_len[i] = static_cast<cl_long>(out.dims[i]);
        params.scale[i] = static_cast<cl_double>(scale_vec[i]);
    }
    params.rank = static_cast<cl_long>(rank);
    params.linear = linear ? 1 : 0;
    params.transform = transform_id;
    params.nearest = nearest_id;

    // One kernel entry for both dtypes: the unused pointer pair is null.
    cl_mem xm = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    const bool ints = in.dtype == DataType::Int64;
    cl_mem xf = ints ? nullptr : xm;
    cl_mem yf = ints ? nullptr : ym;
    cl_mem xl = ints ? xm : nullptr;
    cl_mem yl = ints ? ym : nullptr;
    cl_long use_int64 = ints ? 1 : 0;
    cl_long count = static_cast<cl_long>(out_count);
    cl_kernel kernel = opencl_runtime::KernelFor("Resize", kSource, "resize");
    CheckCl(clSetKernelArg(kernel, 0, sizeof(xf), &xf), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(yf), &yf), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(xl), &xl), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(yl), &yl), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(use_int64), &use_int64), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(count), &count), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(params), &params), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(out_count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
