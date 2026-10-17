// Div on OpenCL: elementwise quotient with NumPy multidirectional broadcasting.
//
// Float division by zero follows IEEE-754 (inf/nan) and never throws; an
// int64 zero divisor is flagged on device and reported on the host, exactly
// like the oracle.
//
// Both inputs share one element type (float32 or int64); the output matches.
// Broadcast strides resolve on the host (same rules as the CPU oracle) into
// rank-indexed arrays; ranks above 16 are rejected like the CUDA pilot.

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

constexpr std::size_t kMaxRank = 16;

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

std::vector<int64_t> BroadcastDims(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                                   const std::string& context) {
    const std::size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank, 1);
    for (std::size_t i = 0; i < rank; ++i) {
        const int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
        const int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
        if (da != db && da != 1 && db != 1) {
            throw InferenceException(context + ": shapes are not broadcast-compatible");
        }
        out[i] = std::max(da, db);
    }
    return out;
}

std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

std::vector<int64_t> AlignedStrides(const std::vector<int64_t>& inDims, const std::vector<int64_t>& outDims,
                                    const std::string& context) {
    const std::size_t outRank = outDims.size();
    const std::size_t inRank = inDims.size();
    const std::vector<int64_t> inStrides = Strides(inDims);
    std::vector<int64_t> aligned(outRank, 0);
    for (std::size_t j = 0; j < outRank; ++j) {
        if (j + inRank < outRank) {
            continue;
        }
        const std::size_t k = j - (outRank - inRank);
        if (inDims[k] == outDims[j]) {
            aligned[j] = inStrides[k];
        } else if (inDims[k] == 1) {
            aligned[j] = 0;
        } else {
            throw InferenceException(context + ": shapes are not broadcast-compatible");
        }
    }
    return aligned;
}

struct StrideSet {
    cl_long s[kMaxRank];
};

StrideSet Pack(const std::vector<int64_t>& strides, const std::string& context) {
    if (strides.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the pilot broadcast limit of 16");
    }
    StrideSet packed{};
    for (std::size_t j = 0; j < strides.size(); ++j) {
        packed.s[j] = static_cast<cl_long>(strides[j]);
    }
    return packed;
}

constexpr const char* kSourceFloat = R"CLC(
typedef struct { long s[16]; } StrideSet;
__kernel void div_f(__global const float* x, __global const float* z, __global float* y, long count,
                    StrideSet outStrides, StrideSet aStrides, StrideSet bStrides, int rank) {
    long i = get_global_id(0);
    if (i < count) {
        long remaining = i;
        long aOffset = 0;
        long bOffset = 0;
        for (int j = 0; j < rank; ++j) {
            long coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            aOffset += coord * aStrides.s[j];
            bOffset += coord * bStrides.s[j];
        }
        y[i] = x[aOffset] / z[bOffset];
    }
}
)CLC";

constexpr const char* kSourceInt = R"CLC(
typedef struct { long s[16]; } StrideSet;
__kernel void div_i(__global const long* x, __global const long* z, __global long* y, long count,
                    StrideSet outStrides, StrideSet aStrides, StrideSet bStrides, int rank,
                    __global int* zeroFlag) {
    long i = get_global_id(0);
    if (i < count) {
        long remaining = i;
        long aOffset = 0;
        long bOffset = 0;
        for (int j = 0; j < rank; ++j) {
            long coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            aOffset += coord * aStrides.s[j];
            bOffset += coord * bStrides.s[j];
        }
        if (z[bOffset] == 0) {
            atomic_xchg(zeroFlag, 1);
            y[i] = 0;
        } else {
            y[i] = x[aOffset] / z[bOffset];
        }
    }
}
)CLC";

void Launch(cl_kernel kernel, cl_mem x, cl_mem z, cl_mem y, int64_t outCount, const StrideSet& outStrides,
            const StrideSet& aStrides, const StrideSet& bStrides, int rank, const std::string& context) {
    cl_long count = static_cast<cl_long>(outCount);
    cl_int rankArg = static_cast<cl_int>(rank);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(z), &z), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(count), &count), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(outStrides), &outStrides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(aStrides), &aStrides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(bStrides), &bStrides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 7, sizeof(rankArg), &rankArg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(outCount);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

void LaunchDivI(cl_kernel kernel, cl_mem x, cl_mem z, cl_mem y, int64_t outCount, const StrideSet& outStrides,
                const StrideSet& aStrides, const StrideSet& bStrides, int rank, const std::string& context) {
    cl_long count = static_cast<cl_long>(outCount);
    cl_int rankArg = static_cast<cl_int>(rank);
    cl_int status = CL_SUCCESS;
    cl_mem flag = clCreateBuffer(opencl_runtime::Context(), CL_MEM_READ_WRITE, sizeof(cl_int), nullptr, &status);
    CheckCl(status, "clCreateBuffer", context);
    const cl_int zero = 0;
    CheckCl(clEnqueueWriteBuffer(opencl_runtime::Queue(), flag, CL_TRUE, 0, sizeof(zero), &zero, 0, nullptr, nullptr),
            "clEnqueueWriteBuffer", context);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(z), &z), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(count), &count), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(outStrides), &outStrides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(aStrides), &aStrides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(bStrides), &bStrides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 7, sizeof(rankArg), &rankArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 8, sizeof(flag), &flag), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(outCount);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
    cl_int flagged = 0;
    CheckCl(
        clEnqueueReadBuffer(opencl_runtime::Queue(), flag, CL_TRUE, 0, sizeof(flagged), &flagged, 0, nullptr, nullptr),
        "clEnqueueReadBuffer", context);
    CheckCl(clReleaseMemObject(flag), "clReleaseMemObject", context);
    if (flagged != 0) {
        throw InferenceException(context + ": division by zero");
    }
}

} // namespace

void Div(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Div (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two inputs and one output");
    }
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView& out = outputs[0];
    if (a.dtype != b.dtype || (a.dtype != DataType::Float32 && a.dtype != DataType::Int64) || out.dtype != a.dtype) {
        throw InferenceException(context + ": tensors must share one float32 or int64 dtype");
    }
    const int64_t outCount = ElementCount(out.dims, context);
    (void)ElementCount(a.dims, context);
    (void)ElementCount(b.dims, context);
    if (BroadcastDims(a.dims, b.dims, context) != out.dims) {
        throw InferenceException(context + ": output shape does not match the broadcast of the inputs");
    }
    if (a.data == nullptr || b.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (outCount == 0) {
        return;
    }
    if (out.dims.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the pilot broadcast limit of 16");
    }
    const StrideSet outStrides = Pack(Strides(out.dims), context);
    const StrideSet aStrides = Pack(AlignedStrides(a.dims, out.dims, context), context);
    const StrideSet bStrides = Pack(AlignedStrides(b.dims, out.dims, context), context);
    const int rank = static_cast<int>(out.dims.size());
    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(a.data));
    cl_mem z = reinterpret_cast<cl_mem>(const_cast<void*>(b.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    if (a.dtype == DataType::Float32) {
        Launch(opencl_runtime::KernelFor("DivF", kSourceFloat, "div_f"), x, z, y, outCount, outStrides, aStrides,
               bStrides, rank, context);
    } else {
        LaunchDivI(opencl_runtime::KernelFor("DivI", kSourceInt, "div_i"), x, z, y, outCount, outStrides, aStrides,
                   bStrides, rank, context);
    }
}

} // namespace opencl_kernels
} // namespace nert
