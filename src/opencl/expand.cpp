// Expand on OpenCL: broadcast the input to a runtime int64 target shape.
//
// Mirrors the CPU oracle: symmetric max-broadcast with NumPy right
// alignment, data in float32/int64/bool, target shapes positive, extra
// leading axes broadcast as size one. The shape operand lives in device
// memory and is read back once (a few int64s, one blocking read). When the
// element counts match, expansion is an identity copy; otherwise one
// work-item computes each output element through zero strides on the
// broadcast axes.

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

std::size_t ElementSize(DataType dtype) {
    return dtype == DataType::Float32 ? sizeof(float) : dtype == DataType::Int64 ? sizeof(int64_t) : sizeof(bool);
}

constexpr const char* kSource = R"CLC(
typedef struct { long s[16]; } StrideSet;
__kernel void expand_f(__global const float* x, __global float* y, long count, StrideSet outStrides,
                       StrideSet inStrides, int rank) {
    long i = get_global_id(0);
    if (i >= count) {
        return;
    }
    long remaining = i;
    long offset = 0;
    for (int j = 0; j < rank; ++j) {
        long coord = remaining / outStrides.s[j];
        remaining %= outStrides.s[j];
        offset += coord * inStrides.s[j];
    }
    y[i] = x[offset];
}
__kernel void expand_i(__global const long* x, __global long* y, long count, StrideSet outStrides,
                       StrideSet inStrides, int rank) {
    long i = get_global_id(0);
    if (i >= count) {
        return;
    }
    long remaining = i;
    long offset = 0;
    for (int j = 0; j < rank; ++j) {
        long coord = remaining / outStrides.s[j];
        remaining %= outStrides.s[j];
        offset += coord * inStrides.s[j];
    }
    y[i] = x[offset];
}
__kernel void expand_b(__global const uchar* x, __global uchar* y, long count, StrideSet outStrides,
                       StrideSet inStrides, int rank) {
    long i = get_global_id(0);
    if (i >= count) {
        return;
    }
    long remaining = i;
    long offset = 0;
    for (int j = 0; j < rank; ++j) {
        long coord = remaining / outStrides.s[j];
        remaining %= outStrides.s[j];
        offset += coord * inStrides.s[j];
    }
    y[i] = x[offset];
}
)CLC";

} // namespace

void Expand(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Expand (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected a data input, a shape input, and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& shape_view = inputs[1];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool) {
        throw InferenceException(context + ": tensors must be float32, int64, or bool");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    if (shape_view.dtype != DataType::Int64) {
        throw InferenceException(context + ": Expand requires a shape input");
    }
    const int64_t target_count = ElementCount(shape_view.dims, context);
    if (target_count > 0 && shape_view.data == nullptr) {
        throw InferenceException(context + ": null shape buffer");
    }
    std::vector<int64_t> target(static_cast<std::size_t>(target_count), 0);
    if (target_count > 0) {
        CheckCl(clEnqueueReadBuffer(
                    opencl_runtime::Queue(), reinterpret_cast<cl_mem>(const_cast<void*>(shape_view.data)), CL_TRUE, 0,
                    static_cast<std::size_t>(target_count) * sizeof(int64_t), target.data(), 0, nullptr, nullptr),
                "clEnqueueReadBuffer", context);
    }

    const std::size_t rank = std::max(in.dims.size(), target.size());
    std::vector<int64_t> expected(rank, 1);
    for (std::size_t i = 0; i < rank; ++i) {
        const int64_t di = static_cast<int64_t>(in.dims.size()) - static_cast<int64_t>(rank) + static_cast<int64_t>(i);
        const int64_t si = static_cast<int64_t>(target.size()) - static_cast<int64_t>(rank) + static_cast<int64_t>(i);
        const int64_t d = di < 0 ? 1 : in.dims[static_cast<std::size_t>(di)];
        if (si < 0) {
            expected[i] = d;
            continue;
        }
        const int64_t s = target[static_cast<std::size_t>(si)];
        if (s < 1) {
            throw InferenceException(context + ": Expand target shape must be positive");
        }
        if (d == s) {
            expected[i] = s;
        } else if (d == 1 || di < 0) {
            expected[i] = s;
        } else if (s == 1) {
            expected[i] = d;
        } else {
            throw InferenceException(context + ": Expand input is not broadcastable to the target shape");
        }
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the expanded shape");
    }

    const int64_t out_count = ElementCount(out.dims, context);
    const int64_t in_count = ElementCount(in.dims, context);
    if (out_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    if (out_count == 0) {
        return;
    }

    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    const int64_t bytes = out_count * static_cast<int64_t>(ElementSize(in.dtype));
    if (in_count == out_count) {
        CheckCl(clEnqueueCopyBuffer(opencl_runtime::Queue(), x, y, 0, 0, static_cast<std::size_t>(bytes), 0, nullptr,
                                    nullptr),
                "clEnqueueCopyBuffer", context);
        CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
        return;
    }

    const std::vector<int64_t> out_strides = Strides(out.dims);
    const std::vector<int64_t> in_strides = Strides(in.dims);
    const int64_t out_rank = static_cast<int64_t>(rank);
    const int64_t in_rank = static_cast<int64_t>(in.dims.size());
    std::vector<int64_t> aligned(static_cast<std::size_t>(out_rank), 0);
    for (int64_t j = 0; j < out_rank; ++j) {
        const int64_t k = j - (out_rank - in_rank);
        if (k < 0 || in.dims[static_cast<std::size_t>(k)] == 1) {
            aligned[static_cast<std::size_t>(j)] = 0;
        } else {
            aligned[static_cast<std::size_t>(j)] = in_strides[static_cast<std::size_t>(k)];
        }
    }
    const StrideSet out_set = Pack(out_strides, context);
    const StrideSet in_set = Pack(aligned, context);
    const cl_int rank_arg = static_cast<cl_int>(out_rank);
    const cl_long count_arg = out_count;
    const char* entry = in.dtype == DataType::Float32 ? "expand_f"
                        : in.dtype == DataType::Int64 ? "expand_i"
                                                      : "expand_b";
    // KernelFor caches per op key; distinct keys keep the dtype-specific
    // entry points from reusing the first compiled kernel.
    const char* cache_key = in.dtype == DataType::Float32 ? "ExpandF"
                            : in.dtype == DataType::Int64 ? "ExpandI"
                                                          : "ExpandB";
    cl_kernel kernel = opencl_runtime::KernelFor(cache_key, kSource, entry);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(count_arg), &count_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(out_set), &out_set), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(in_set), &in_set), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(rank_arg), &rank_arg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(out_count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
