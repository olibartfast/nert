// Comparison and logical rows on OpenCL: Equal, Greater, LessOrEqual, And.
//
// Mirrors the CPU oracles: two inputs with NumPy broadcast, bool output;
// Equality/ordering take one shared float32 or int64 dtype, And takes bool.
// One work-item per output with zero strides on broadcast axes.

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

std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
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
    for (std::size_t i = 0; i < strides.size(); ++i) {
        packed.s[i] = strides[i];
    }
    return packed;
}

constexpr const char* kSource = R"CLC(
typedef struct { long s[16]; } StrideSet;
#define DECODE_OFFSETS(c) \
    long i = get_global_id(0); \
    if (i >= count) { return; } \
    long remaining = i, ao = 0, bo = 0; \
    for (int j = 0; j < rank; ++j) { \
        long coord = remaining / outStrides.s[j]; \
        remaining %= outStrides.s[j]; \
        ao += coord * aStrides.s[j]; \
        bo += coord * bStrides.s[j]; \
    }
__kernel void equal_f(__global const float* a, __global const float* b, __global uchar* y, long count,
                      StrideSet outStrides, StrideSet aStrides, StrideSet bStrides, int rank) {
    DECODE_OFFSETS(c)
    y[i] = a[ao] == b[bo] ? 1 : 0;
}
__kernel void equal_i(__global const long* a, __global const long* b, __global uchar* y, long count,
                      StrideSet outStrides, StrideSet aStrides, StrideSet bStrides, int rank) {
    DECODE_OFFSETS(c)
    y[i] = a[ao] == b[bo] ? 1 : 0;
}
__kernel void greater_f(__global const float* a, __global const float* b, __global uchar* y, long count,
                        StrideSet outStrides, StrideSet aStrides, StrideSet bStrides, int rank) {
    DECODE_OFFSETS(c)
    y[i] = a[ao] > b[bo] ? 1 : 0;
}
__kernel void greater_i(__global const long* a, __global const long* b, __global uchar* y, long count,
                        StrideSet outStrides, StrideSet aStrides, StrideSet bStrides, int rank) {
    DECODE_OFFSETS(c)
    y[i] = a[ao] > b[bo] ? 1 : 0;
}
__kernel void less_equal_f(__global const float* a, __global const float* b, __global uchar* y, long count,
                           StrideSet outStrides, StrideSet aStrides, StrideSet bStrides, int rank) {
    DECODE_OFFSETS(c)
    y[i] = a[ao] <= b[bo] ? 1 : 0;
}
__kernel void less_equal_i(__global const long* a, __global const long* b, __global uchar* y, long count,
                           StrideSet outStrides, StrideSet aStrides, StrideSet bStrides, int rank) {
    DECODE_OFFSETS(c)
    y[i] = a[ao] <= b[bo] ? 1 : 0;
}
__kernel void and_b(__global const uchar* a, __global const uchar* b, __global uchar* y, long count,
                    StrideSet outStrides, StrideSet aStrides, StrideSet bStrides, int rank) {
    DECODE_OFFSETS(c)
    y[i] = (a[ao] != 0 && b[bo] != 0) ? 1 : 0;
}
)CLC";

void Run(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs,
         const std::string& op_name, bool logical_and) {
    const std::string context = op_name + " (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two inputs and one output");
    }
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView& out = outputs[0];
    if (logical_and) {
        if (a.dtype != DataType::Bool || b.dtype != DataType::Bool) {
            throw InferenceException(context + ": inputs must be bool");
        }
    } else if (a.dtype != b.dtype || (a.dtype != DataType::Float32 && a.dtype != DataType::Int64)) {
        throw InferenceException(context + ": inputs must share one float32 or int64 dtype");
    }
    if (out.dtype != DataType::Bool) {
        throw InferenceException(context + ": output must be bool");
    }
    const int64_t out_count = ElementCount(out.dims, context);
    (void)ElementCount(a.dims, context);
    (void)ElementCount(b.dims, context);
    if (BroadcastDims(a.dims, b.dims, context) != out.dims) {
        throw InferenceException(context + ": output shape does not match the broadcast of the inputs");
    }
    if (a.data == nullptr || b.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (out_count == 0) {
        return;
    }
    const StrideSet out_strides = Pack(Strides(out.dims), context);
    const StrideSet a_strides = Pack(AlignedStrides(a.dims, out.dims, context), context);
    const StrideSet b_strides = Pack(AlignedStrides(b.dims, out.dims, context), context);
    const cl_int rank = static_cast<cl_int>(out.dims.size());
    const cl_long count = out_count;
    cl_mem am = reinterpret_cast<cl_mem>(const_cast<void*>(a.data));
    cl_mem bm = reinterpret_cast<cl_mem>(const_cast<void*>(b.data));
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);

    const char* entry = nullptr;
    const char* cache_key = nullptr;
    if (logical_and) {
        entry = "and_b";
        cache_key = "AndB";
    } else if (op_name == "Equal") {
        entry = a.dtype == DataType::Float32 ? "equal_f" : "equal_i";
        cache_key = a.dtype == DataType::Float32 ? "EqualF" : "EqualI";
    } else if (op_name == "Greater") {
        entry = a.dtype == DataType::Float32 ? "greater_f" : "greater_i";
        cache_key = a.dtype == DataType::Float32 ? "GreaterF" : "GreaterI";
    } else {
        entry = a.dtype == DataType::Float32 ? "less_equal_f" : "less_equal_i";
        cache_key = a.dtype == DataType::Float32 ? "LessEqualF" : "LessEqualI";
    }
    cl_kernel kernel = opencl_runtime::KernelFor(cache_key, kSource, entry);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(am), &am), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(bm), &bm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(count), &count), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(out_strides), &out_strides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(a_strides), &a_strides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(b_strides), &b_strides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 7, sizeof(rank), &rank), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(out_count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace

void Equal(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    Run(node, inputs, outputs, "Equal", false);
}

void Greater(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    Run(node, inputs, outputs, "Greater", false);
}

void LessOrEqual(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    Run(node, inputs, outputs, "LessOrEqual", false);
}

void And(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    Run(node, inputs, outputs, "And", true);
}

} // namespace opencl_kernels
} // namespace nert
