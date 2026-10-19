// GridSample on OpenCL: sample a 4-D float32 input at normalized positions.
//
// Mirrors the CPU oracle exactly: X is [N,C,H,W] and grid is [N,Ho,Wo,2]
// with (x, y) in normalized coordinates (x indexes W, y indexes H); the
// output is [N,C,Ho,Wo]. Attributes: mode (bilinear or nearest, default
// bilinear), padding_mode (zeros, border, reflection, default zeros) and
// align_corners (0 or 1, default 0). The align_corners denormalization and
// the padding run in double (the device must report double-precision
// support, cl_khr_fp64), padding is applied to the continuous coordinate,
// bilinear is the 4-tap weighted average and nearest rounds half to even,
// matching the oracle's std::nearbyint. One work-item per output element
// with 64-bit index math; the OpenCL C source compiles once per process
// via the shared runtime.

#include "nert/device.hpp"
#include "runtime.hpp"

#include <cstdint>
#include <string>
#include <variant>
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

std::string StringAttribute(const Node& node, const std::string& name, const std::string& fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::string* value = std::get_if<std::string>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("GridSample: attribute '" + name + "' must be a string (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("GridSample: attribute '" + name + "' must be an int (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

// Padding encodings shared with the kernel below: 0 zeros, 1 border,
// 2 reflection.

constexpr const char* kSource = R"CLC(
// Value at (yy, xx) of a plane, zero outside (the zeros padding for the
// taps; border and reflection keep the coordinates in range).
static double at(__global const float* plane, long yy, long xx, long h, long w) {
    if (yy < 0 || yy >= h || xx < 0 || xx >= w) {
        return 0.0;
    }
    return (double)plane[yy * w + xx];
}

// Reflect x about [lo, hi] (the border coordinates), exactly as the oracle.
static double reflect(double x, double lo, double hi) {
    if (hi == lo) {
        return lo;
    }
    const double range = hi - lo;
    const double d = fabs(x - lo);
    const double flips = floor(d / range);
    const double extra = d - flips * range;
    return fmod(flips, 2.0) == 0.0 ? lo + extra : hi - extra;
}

// Normalized coordinate -> source pixel coordinate on an axis of `len`
// pixels, with the padding mode applied to the continuous coordinate.
static double source_coordinate(double g, long len, int align_corners, int padding) {
    const double size = (double)len;
    double x = align_corners ? (g + 1.0) / 2.0 * (size - 1.0) : ((g + 1.0) * size - 1.0) / 2.0;
    if (padding == 2) {
        x = align_corners ? reflect(x, 0.0, size - 1.0) : reflect(x, -0.5, size - 0.5);
        x = fmin(fmax(x, 0.0), size - 1.0);
    } else if (padding == 1) {
        x = fmin(fmax(x, 0.0), size - 1.0);
    }
    return x;
}

// One work-item per output element; the output is indexed as
// ((b * c + ch) * ho + i) * wo + j.
__kernel void grid_sample(__global const float* x, __global const float* grid, __global float* y, long n, long c,
                          long h, long w, long ho, long wo, int nearest, int padding, int align_corners) {
    const long total = n * c * ho * wo;
    const long index = (long)get_global_id(0);
    if (index < total) {
        const long j = index % wo;
        long t = index / wo;
        const long i = t % ho;
        t /= ho;
        const long ch = t % c;
        const long b = t / c;
        __global const float* g = grid + ((b * ho + i) * wo + j) * 2;
        const double px = source_coordinate((double)g[0], w, align_corners, padding);
        const double py = source_coordinate((double)g[1], h, align_corners, padding);
        __global const float* plane = x + (b * c + ch) * h * w;
        double value = 0.0;
        if (nearest != 0) {
            // rint rounds half to even, like the oracle's std::nearbyint.
            const long yy = (long)rint(py);
            const long xx = (long)rint(px);
            if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
                value = (double)plane[yy * w + xx];
            }
        } else {
            const double fx = floor(px);
            const double fy = floor(py);
            const long x0 = (long)fx;
            const long y0 = (long)fy;
            const double wx = px - fx;
            const double wy = py - fy;
            value = at(plane, y0, x0, h, w) * (1.0 - wx) * (1.0 - wy) + at(plane, y0, x0 + 1, h, w) * wx * (1.0 - wy) +
                    at(plane, y0 + 1, x0, h, w) * (1.0 - wx) * wy + at(plane, y0 + 1, x0 + 1, h, w) * wx * wy;
        }
        y[index] = (float)value;
    }
}
)CLC";

} // namespace

void GridSample(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "GridSample (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two inputs and one output");
    }
    const std::string mode = StringAttribute(node, "mode", "bilinear");
    const std::string padding = StringAttribute(node, "padding_mode", "zeros");
    const int64_t alignAttr = IntAttribute(node, "align_corners", 0);
    if (mode != "bilinear" && mode != "nearest") {
        throw InferenceException(context + ": unsupported mode '" + mode + "'");
    }
    if (padding != "zeros" && padding != "border" && padding != "reflection") {
        throw InferenceException(context + ": unsupported padding_mode '" + padding + "'");
    }
    if (alignAttr != 0 && alignAttr != 1) {
        throw InferenceException(context + ": align_corners must be 0 or 1");
    }
    const bool alignCorners = alignAttr == 1;

    const TensorView& x = inputs[0];
    const TensorView& grid = inputs[1];
    const TensorView& out = outputs[0];
    if (x.dtype != DataType::Float32 || grid.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (x.dims.size() != 4 || grid.dims.size() != 4 || grid.dims[3] != 2) {
        throw InferenceException(context + ": expected X [N,C,H,W] and grid [N,Ho,Wo,2]");
    }
    const int64_t n = x.dims[0];
    const int64_t c = x.dims[1];
    const int64_t h = x.dims[2];
    const int64_t w = x.dims[3];
    const int64_t ho = grid.dims[1];
    const int64_t wo = grid.dims[2];
    if (grid.dims[0] != n) {
        throw InferenceException(context + ": batch sizes of X and grid differ");
    }
    if (out.dims != std::vector<int64_t>({n, c, ho, wo})) {
        throw InferenceException(context + ": output shape does not match the operands");
    }
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || ho <= 0 || wo <= 0) {
        throw InferenceException(context + ": non-positive dimension");
    }
    if (x.data == nullptr || grid.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    // One work-item per output element. The positive-dimension checks above
    // already reject zero extents, so this is a defensive guard: never
    // launch an empty range.
    const int64_t count = n * c * ho * wo;
    if (count == 0) {
        return;
    }
    // The kernel computes coordinates in double; OpenCL fp64 is optional,
    // so fail with the device capability named instead of a build error.
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
    cl_mem gm = reinterpret_cast<cl_mem>(const_cast<void*>(grid.data));
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    cl_long nArg = static_cast<cl_long>(n);
    cl_long cArg = static_cast<cl_long>(c);
    cl_long hArg = static_cast<cl_long>(h);
    cl_long wArg = static_cast<cl_long>(w);
    cl_long hoArg = static_cast<cl_long>(ho);
    cl_long woArg = static_cast<cl_long>(wo);
    cl_int nearestArg = mode == "nearest" ? 1 : 0;
    cl_int paddingArg = padding == "zeros" ? 0 : (padding == "border" ? 1 : 2);
    cl_int alignArg = alignCorners ? 1 : 0;
    cl_kernel kernel = opencl_runtime::KernelFor("GridSample", kSource, "grid_sample");
    CheckCl(clSetKernelArg(kernel, 0, sizeof(xm), &xm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(gm), &gm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(nArg), &nArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(cArg), &cArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(hArg), &hArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(wArg), &wArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 7, sizeof(hoArg), &hoArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 8, sizeof(woArg), &woArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 9, sizeof(nearestArg), &nearestArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 10, sizeof(paddingArg), &paddingArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 11, sizeof(alignArg), &alignArg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
