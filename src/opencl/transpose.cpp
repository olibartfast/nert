// Transpose on OpenCL: permuted copy, one work-item per output element.
//
// Mirrors the CPU oracle: float32/int64/bool sharing one dtype, perm attr
// defaulting to the reversed rank, permutation + output-shape checks. The
// OpenCL C source compiles once per process via the shared runtime. Ranks
// above 16 are rejected (pilot limit, shared with Add).

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

struct IndexSet {
    cl_long s[kMaxRank];
};

IndexSet Pack(const std::vector<int64_t>& values, const std::string& context) {
    if (values.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the transpose limit of 16");
    }
    IndexSet packed{};
    for (std::size_t j = 0; j < values.size(); ++j) {
        packed.s[j] = static_cast<cl_long>(values[j]);
    }
    return packed;
}

constexpr const char* kSourceFloat = R"CLC(
typedef struct { long s[16]; } StrideSet;
__kernel void transpose_f(__global const float* x, __global float* y, long count, StrideSet perm,
                          StrideSet inStrides, StrideSet outStrides, int rank) {
    long i = get_global_id(0);
    if (i < count) {
        long remaining = i;
        long offset = 0;
        for (int j = 0; j < rank; ++j) {
            long coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            offset += coord * inStrides.s[perm.s[j]];
        }
        y[i] = x[offset];
    }
}
)CLC";

constexpr const char* kSourceInt = R"CLC(
typedef struct { long s[16]; } StrideSet;
__kernel void transpose_i(__global const long* x, __global long* y, long count, StrideSet perm,
                          StrideSet inStrides, StrideSet outStrides, int rank) {
    long i = get_global_id(0);
    if (i < count) {
        long remaining = i;
        long offset = 0;
        for (int j = 0; j < rank; ++j) {
            long coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            offset += coord * inStrides.s[perm.s[j]];
        }
        y[i] = x[offset];
    }
}
)CLC";

constexpr const char* kSourceBool = R"CLC(
typedef struct { long s[16]; } StrideSet;
__kernel void transpose_b(__global const uchar* x, __global uchar* y, long count, StrideSet perm,
                          StrideSet inStrides, StrideSet outStrides, int rank) {
    long i = get_global_id(0);
    if (i < count) {
        long remaining = i;
        long offset = 0;
        for (int j = 0; j < rank; ++j) {
            long coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            offset += coord * inStrides.s[perm.s[j]];
        }
        y[i] = x[offset];
    }
}
)CLC";

void Launch(cl_kernel kernel, cl_mem x, cl_mem y, int64_t outCount, const IndexSet& perm, const IndexSet& inStrides,
            const IndexSet& outStrides, int rank, const std::string& context) {
    cl_long count = static_cast<cl_long>(outCount);
    cl_int rankArg = static_cast<cl_int>(rank);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(count), &count), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(perm), &perm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(inStrides), &inStrides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(outStrides), &outStrides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(rankArg), &rankArg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(outCount);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace

void Transpose(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Transpose (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool) {
        throw InferenceException(context + ": tensors must be float32, int64, or bool");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (static_cast<std::size_t>(rank) > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the transpose limit of 16");
    }

    std::vector<int64_t> perm;
    bool have_perm = false;
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "perm") {
            const std::vector<int64_t>* value = std::get_if<std::vector<int64_t>>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'perm' must be an int list");
            }
            perm = *value;
            have_perm = true;
        }
    }
    if (!have_perm) {
        perm.resize(static_cast<std::size_t>(rank));
        for (int64_t i = 0; i < rank; ++i) {
            perm[static_cast<std::size_t>(i)] = rank - 1 - i;
        }
    }
    if (static_cast<int64_t>(perm.size()) != rank) {
        throw InferenceException(context + ": Transpose perm rank must match the input rank");
    }
    std::vector<int64_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (int64_t i = 0; i < rank; ++i) {
        if (sorted[static_cast<std::size_t>(i)] != i) {
            throw InferenceException(context + ": Transpose perm must be a permutation");
        }
    }
    std::vector<int64_t> expected(static_cast<std::size_t>(rank), 0);
    for (int64_t i = 0; i < rank; ++i) {
        expected[static_cast<std::size_t>(i)] = in.dims[static_cast<std::size_t>(perm[i])];
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the transposed input");
    }

    const int64_t count = ElementCount(out.dims, context);
    (void)ElementCount(in.dims, context);
    if (count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    if (count == 0) {
        return;
    }

    const IndexSet permSet = Pack(perm, context);
    const IndexSet inStrides = Pack(Strides(in.dims), context);
    const IndexSet outStrides = Pack(Strides(out.dims), context);
    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    if (in.dtype == DataType::Float32) {
        Launch(opencl_runtime::KernelFor("TransposeF", kSourceFloat, "transpose_f"), x, y, count, permSet, inStrides,
               outStrides, static_cast<int>(rank), context);
    } else if (in.dtype == DataType::Int64) {
        Launch(opencl_runtime::KernelFor("TransposeI", kSourceInt, "transpose_i"), x, y, count, permSet, inStrides,
               outStrides, static_cast<int>(rank), context);
    } else {
        Launch(opencl_runtime::KernelFor("TransposeB", kSourceBool, "transpose_b"), x, y, count, permSet, inStrides,
               outStrides, static_cast<int>(rank), context);
    }
}

} // namespace opencl_kernels
} // namespace nert
