// Where on OpenCL: broadcast ternary select, one work-item per output.
//
// Mirrors the CPU oracle: bool condition, f32/i64 values sharing one dtype
// with the output, three-way NumPy broadcast. Same stride template and rank
// cap (16) as Add; the bool condition is read one byte per element.

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
__kernel void where_f(__global const uchar* c, __global const float* x, __global const float* z,
                      __global float* y, long count, StrideSet outStrides, StrideSet cStrides,
                      StrideSet xStrides, StrideSet zStrides, int rank) {
    long i = get_global_id(0);
    if (i < count) {
        long remaining = i;
        long cOffset = 0;
        long xOffset = 0;
        long zOffset = 0;
        for (int j = 0; j < rank; ++j) {
            long coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            cOffset += coord * cStrides.s[j];
            xOffset += coord * xStrides.s[j];
            zOffset += coord * zStrides.s[j];
        }
        y[i] = c[cOffset] != 0 ? x[xOffset] : z[zOffset];
    }
}
)CLC";

constexpr const char* kSourceInt = R"CLC(
typedef struct { long s[16]; } StrideSet;
__kernel void where_i(__global const uchar* c, __global const long* x, __global const long* z,
                      __global long* y, long count, StrideSet outStrides, StrideSet cStrides,
                      StrideSet xStrides, StrideSet zStrides, int rank) {
    long i = get_global_id(0);
    if (i < count) {
        long remaining = i;
        long cOffset = 0;
        long xOffset = 0;
        long zOffset = 0;
        for (int j = 0; j < rank; ++j) {
            long coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            cOffset += coord * cStrides.s[j];
            xOffset += coord * xStrides.s[j];
            zOffset += coord * zStrides.s[j];
        }
        y[i] = c[cOffset] != 0 ? x[xOffset] : z[zOffset];
    }
}
)CLC";

void Launch(cl_kernel kernel, cl_mem c, cl_mem x, cl_mem z, cl_mem y, int64_t outCount, const StrideSet& outStrides,
            const StrideSet& cStrides, const StrideSet& xStrides, const StrideSet& zStrides, int rank,
            const std::string& context) {
    cl_long count = static_cast<cl_long>(outCount);
    cl_int rankArg = static_cast<cl_int>(rank);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(c), &c), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(z), &z), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(count), &count), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(outStrides), &outStrides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(cStrides), &cStrides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 7, sizeof(xStrides), &xStrides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 8, sizeof(zStrides), &zStrides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 9, sizeof(rankArg), &rankArg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(outCount);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace

void Where(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Where (node '" + node.name + "')";
    if (inputs.size() != 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected three inputs and one output");
    }
    const TensorView& cond = inputs[0];
    const TensorView& x = inputs[1];
    const TensorView& z = inputs[2];
    const TensorView& out = outputs[0];
    if (cond.dtype != DataType::Bool) {
        throw InferenceException(context + ": condition must be bool");
    }
    if (x.dtype != z.dtype || (x.dtype != DataType::Float32 && x.dtype != DataType::Int64) || out.dtype != x.dtype) {
        throw InferenceException(context + ": value tensors must share one float32 or int64 dtype");
    }

    const int64_t outCount = ElementCount(out.dims, context);
    (void)ElementCount(cond.dims, context);
    (void)ElementCount(x.dims, context);
    (void)ElementCount(z.dims, context);
    if (BroadcastDims(BroadcastDims(cond.dims, x.dims, context), z.dims, context) != out.dims) {
        throw InferenceException(context + ": output shape does not match the broadcast of the inputs");
    }
    if (cond.data == nullptr || x.data == nullptr || z.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (outCount == 0) {
        return;
    }
    if (out.dims.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the pilot broadcast limit of 16");
    }
    const StrideSet outStrides = Pack(Strides(out.dims), context);
    const StrideSet cStrides = Pack(AlignedStrides(cond.dims, out.dims, context), context);
    const StrideSet xStrides = Pack(AlignedStrides(x.dims, out.dims, context), context);
    const StrideSet zStrides = Pack(AlignedStrides(z.dims, out.dims, context), context);
    const int rank = static_cast<int>(out.dims.size());
    const bool isFloat = x.dtype == DataType::Float32;
    cl_mem cm = reinterpret_cast<cl_mem>(const_cast<void*>(cond.data));
    cl_mem xm = reinterpret_cast<cl_mem>(const_cast<void*>(x.data));
    cl_mem zm = reinterpret_cast<cl_mem>(const_cast<void*>(z.data));
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    if (isFloat) {
        Launch(opencl_runtime::KernelFor("WhereF", kSourceFloat, "where_f"), cm, xm, zm, ym, outCount, outStrides,
               cStrides, xStrides, zStrides, rank, context);
    } else {
        Launch(opencl_runtime::KernelFor("WhereI", kSourceInt, "where_i"), cm, xm, zm, ym, outCount, outStrides,
               cStrides, xStrides, zStrides, rank, context);
    }
}

} // namespace opencl_kernels
} // namespace nert
