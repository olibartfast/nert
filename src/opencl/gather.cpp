// Gather on OpenCL: indexed copy along one axis, one work-item per
// output element.
//
// Mirrors the CPU oracle: float32/int64/bool sharing one dtype, int64
// indices, axis attr default 0 with negative wrap, negative index wrap.
// An out-of-range index is flagged on device (atomically, Div precedent)
// and reported on the host, since device code cannot throw. The OpenCL C
// source compiles once per process via the shared runtime.

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

constexpr const char* kSourceFloat = R"CLC(
__kernel void gather_f(__global const float* x, __global const long* idx, __global float* y, long outCount,
                       long indexCount, long inner, long dim, __global int* oob) {
    long i = get_global_id(0);
    if (i < outCount) {
        long row = indexCount * inner;
        long o = i / row;
        long r = i % row;
        long j = r / inner;
        long n = r % inner;
        long k = idx[j];
        if (k < 0) {
            k += dim;
        }
        if (k < 0 || k >= dim) {
            atomic_xchg(oob, 1);
            y[i] = 0.0f;
            return;
        }
        y[i] = x[(o * dim + k) * inner + n];
    }
}
)CLC";

constexpr const char* kSourceInt = R"CLC(
__kernel void gather_i(__global const long* x, __global const long* idx, __global long* y, long outCount,
                       long indexCount, long inner, long dim, __global int* oob) {
    long i = get_global_id(0);
    if (i < outCount) {
        long row = indexCount * inner;
        long o = i / row;
        long r = i % row;
        long j = r / inner;
        long n = r % inner;
        long k = idx[j];
        if (k < 0) {
            k += dim;
        }
        if (k < 0 || k >= dim) {
            atomic_xchg(oob, 1);
            y[i] = 0;
            return;
        }
        y[i] = x[(o * dim + k) * inner + n];
    }
}
)CLC";

constexpr const char* kSourceBool = R"CLC(
__kernel void gather_b(__global const uchar* x, __global const long* idx, __global uchar* y, long outCount,
                       long indexCount, long inner, long dim, __global int* oob) {
    long i = get_global_id(0);
    if (i < outCount) {
        long row = indexCount * inner;
        long o = i / row;
        long r = i % row;
        long j = r / inner;
        long n = r % inner;
        long k = idx[j];
        if (k < 0) {
            k += dim;
        }
        if (k < 0 || k >= dim) {
            atomic_xchg(oob, 1);
            y[i] = 0;
            return;
        }
        y[i] = x[(o * dim + k) * inner + n];
    }
}
)CLC";

void Launch(cl_kernel kernel, cl_mem x, cl_mem idx, cl_mem y, cl_mem oob, int64_t outCount, int64_t indexCount,
            int64_t inner, int64_t dim, const std::string& context) {
    cl_long outC = outCount, indexC = indexCount, inr = inner, d = dim;
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(idx), &idx), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(outC), &outC), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(indexC), &indexC), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(inr), &inr), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(d), &d), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 7, sizeof(oob), &oob), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(outCount);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace

void Gather(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Gather (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected data and indices inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& indices = inputs[1];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool) {
        throw InferenceException(context + ": tensors must be float32, int64, or bool");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    if (indices.dtype != DataType::Int64) {
        throw InferenceException(context + ": Gather indices must be int64");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (rank == 0) {
        throw InferenceException(context + ": Gather axis is out of range");
    }
    int64_t axis = 0;
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "axis") {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'axis' must be an integer");
            }
            axis = *value;
        }
    }
    if (axis < -rank || axis >= rank) {
        throw InferenceException(context + ": Gather axis is out of range");
    }
    if (axis < 0) {
        axis += rank;
    }

    std::vector<int64_t> expected;
    for (int64_t i = 0; i < axis; ++i) {
        expected.push_back(in.dims[static_cast<std::size_t>(i)]);
    }
    expected.insert(expected.end(), indices.dims.begin(), indices.dims.end());
    for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < in.dims.size(); ++i) {
        expected.push_back(in.dims[i]);
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the gathered indices");
    }

    const int64_t index_count = ElementCount(indices.dims, context);
    const int64_t out_count = ElementCount(out.dims, context);
    (void)ElementCount(in.dims, context);
    if (index_count > 0 && indices.data == nullptr) {
        throw InferenceException(context + ": null indices buffer");
    }
    if (out_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    if (out_count == 0) {
        return;
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= in.dims[static_cast<std::size_t>(i)];
    }
    (void)outer;
    const int64_t dim = in.dims[static_cast<std::size_t>(axis)];
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= in.dims[static_cast<std::size_t>(i)];
    }

    cl_int oobError = CL_SUCCESS;
    cl_mem oob = clCreateBuffer(opencl_runtime::Context(), CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &oobError);
    CheckCl(oobError, "clCreateBuffer", context);
    cl_int zero = 0;
    CheckCl(clEnqueueWriteBuffer(opencl_runtime::Queue(), oob, CL_TRUE, 0, sizeof(zero), &zero, 0, nullptr, nullptr),
            "clEnqueueWriteBuffer", context);

    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem idx = reinterpret_cast<cl_mem>(const_cast<void*>(indices.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    if (in.dtype == DataType::Float32) {
        Launch(opencl_runtime::KernelFor("GatherF", kSourceFloat, "gather_f"), x, idx, y, oob, out_count, index_count,
               inner, dim, context);
    } else if (in.dtype == DataType::Int64) {
        Launch(opencl_runtime::KernelFor("GatherI", kSourceInt, "gather_i"), x, idx, y, oob, out_count, index_count,
               inner, dim, context);
    } else {
        Launch(opencl_runtime::KernelFor("GatherB", kSourceBool, "gather_b"), x, idx, y, oob, out_count, index_count,
               inner, dim, context);
    }
    cl_int flagged = 0;
    CheckCl(
        clEnqueueReadBuffer(opencl_runtime::Queue(), oob, CL_TRUE, 0, sizeof(flagged), &flagged, 0, nullptr, nullptr),
        "clEnqueueReadBuffer", context);
    CheckCl(clReleaseMemObject(oob), "clReleaseMemObject", context);
    if (flagged != 0) {
        throw InferenceException(context + ": Gather index is out of range");
    }
}

} // namespace opencl_kernels
} // namespace nert
