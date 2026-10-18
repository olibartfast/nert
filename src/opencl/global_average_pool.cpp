// GlobalAveragePool on OpenCL: the whole-spatial average.
//
// Mirrors the CPU oracle (via the verified CUDA row): rank>=2 input,
// output [N, C, 1, ..., 1], each (n, c) output is the mean of its
// spatial plane accumulated in double with one float32 rounding (the
// device must report double-precision support, cl_khr_fp64). One
// work-item per (n, c) row. The OpenCL C source compiles once per
// process via the shared runtime.

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

constexpr const char* kSource = R"CLC(
__kernel void gap(__global const float* x, __global float* y, long rows, long spatial) {
    long r = get_global_id(0);
    if (r < rows) {
        __global const float* row = x + r * spatial;
        double total = 0.0;
        for (long k = 0; k < spatial; ++k) {
            total += (double)row[k];
        }
        y[r] = (float)(total / (double)spatial);
    }
}
)CLC";

} // namespace

void GlobalAveragePool(const Node& node, const std::vector<TensorView>& inputs,
                       const std::vector<TensorView>& outputs) {
    const std::string context = "GlobalAveragePool (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& x = inputs[0];
    const TensorView& out = outputs[0];
    if (x.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (x.dims.size() < 2) {
        throw InferenceException(context + ": input must have batch and channel axes (rank at least 2)");
    }
    (void)ElementCount(x.dims, context);
    (void)ElementCount(out.dims, context);

    std::vector<int64_t> expected = {x.dims[0], x.dims[1]};
    expected.insert(expected.end(), x.dims.size() - 2, 1);
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape must be the input shape with all spatial axes at 1");
    }
    if (x.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const int64_t rows = x.dims[0] * x.dims[1];
    int64_t spatial = 1;
    for (std::size_t i = 2; i < x.dims.size(); ++i) {
        spatial *= x.dims[i];
    }
    if (rows == 0) {
        return;
    }
    // The kernel accumulates in double; OpenCL fp64 is optional, so fail
    // with the device capability named instead of a build error.
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
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    cl_long rowsArg = static_cast<cl_long>(rows);
    cl_long spatialArg = static_cast<cl_long>(spatial);
    cl_kernel kernel = opencl_runtime::KernelFor("GlobalAveragePool", kSource, "gap");
    CheckCl(clSetKernelArg(kernel, 0, sizeof(xm), &xm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(rowsArg), &rowsArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(spatialArg), &spatialArg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(rows);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
