// Slice on OpenCL: strided sub-tensor copy, one work-item per output
// element.
//
// Mirrors the CPU oracle: float32/int64/bool sharing one dtype,
// starts/ends (+ optional axes/steps) operands downloaded from device,
// negative wrap, clamping, positive and negative steps, empty selections
// carrying only the shape contract. The OpenCL C source compiles once per
// process via the shared runtime. Ranks above 16 are rejected (pilot
// limit, shared with Add).

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

std::vector<int64_t> ReadInt64s(const TensorView& view, const std::string& what, const std::string& context) {
    if (view.dtype != DataType::Int64) {
        throw InferenceException(context + ": Slice " + what + " must be int64");
    }
    const int64_t count = ElementCount(view.dims, context);
    if (count > 0 && view.data == nullptr) {
        throw InferenceException(context + ": null " + what + " buffer");
    }
    std::vector<int64_t> values(static_cast<std::size_t>(count), 0);
    if (count > 0) {
        CheckCl(clEnqueueReadBuffer(opencl_runtime::Queue(), reinterpret_cast<cl_mem>(const_cast<void*>(view.data)),
                                    CL_TRUE, 0, static_cast<std::size_t>(count) * sizeof(int64_t), values.data(), 0,
                                    nullptr, nullptr),
                "clEnqueueReadBuffer", context);
    }
    return values;
}

struct IndexSet {
    cl_long s[kMaxRank];
};

IndexSet Pack(const std::vector<int64_t>& values, const std::string& context) {
    if (values.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the slice limit of 16");
    }
    IndexSet packed{};
    for (std::size_t j = 0; j < values.size(); ++j) {
        packed.s[j] = static_cast<cl_long>(values[j]);
    }
    return packed;
}

constexpr const char* kSourceFloat = R"CLC(
typedef struct { long s[16]; } StrideSet;
__kernel void slice_f(__global const float* x, __global float* y, long count, StrideSet begins, StrideSet steps,
                      StrideSet inStrides, StrideSet outStrides, int rank) {
    long i = get_global_id(0);
    if (i < count) {
        long remaining = i;
        long offset = 0;
        for (int j = 0; j < rank; ++j) {
            long coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            offset += (begins.s[j] + coord * steps.s[j]) * inStrides.s[j];
        }
        y[i] = x[offset];
    }
}
)CLC";

constexpr const char* kSourceInt = R"CLC(
typedef struct { long s[16]; } StrideSet;
__kernel void slice_i(__global const long* x, __global long* y, long count, StrideSet begins, StrideSet steps,
                      StrideSet inStrides, StrideSet outStrides, int rank) {
    long i = get_global_id(0);
    if (i < count) {
        long remaining = i;
        long offset = 0;
        for (int j = 0; j < rank; ++j) {
            long coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            offset += (begins.s[j] + coord * steps.s[j]) * inStrides.s[j];
        }
        y[i] = x[offset];
    }
}
)CLC";

constexpr const char* kSourceBool = R"CLC(
typedef struct { long s[16]; } StrideSet;
__kernel void slice_b(__global const uchar* x, __global uchar* y, long count, StrideSet begins, StrideSet steps,
                      StrideSet inStrides, StrideSet outStrides, int rank) {
    long i = get_global_id(0);
    if (i < count) {
        long remaining = i;
        long offset = 0;
        for (int j = 0; j < rank; ++j) {
            long coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            offset += (begins.s[j] + coord * steps.s[j]) * inStrides.s[j];
        }
        y[i] = x[offset];
    }
}
)CLC";

void Launch(cl_kernel kernel, cl_mem x, cl_mem y, const IndexSet& begins, const IndexSet& steps,
            const IndexSet& inStrides, const IndexSet& outStrides, int64_t outCount, int rank,
            const std::string& context) {
    cl_long count = static_cast<cl_long>(outCount);
    cl_int rankArg = static_cast<cl_int>(rank);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(count), &count), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(begins), &begins), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(steps), &steps), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(inStrides), &inStrides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(outStrides), &outStrides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 7, sizeof(rankArg), &rankArg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(outCount);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace

void Slice(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Slice (node '" + node.name + "')";
    if (inputs.size() < 3 || inputs.size() > 5 || outputs.size() != 1) {
        throw InferenceException(context + ": expected data, starts, ends, and optional axes/steps inputs");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool) {
        throw InferenceException(context + ": tensors must be float32, int64, or bool");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    const std::vector<int64_t> starts = ReadInt64s(inputs[1], "starts", context);
    const std::vector<int64_t> ends = ReadInt64s(inputs[2], "ends", context);
    if (starts.size() != ends.size()) {
        throw InferenceException(context + ": Slice starts and ends must have the same length");
    }
    const std::size_t count = starts.size();

    std::vector<int64_t> axes(count, 0);
    for (std::size_t i = 0; i < count; ++i) {
        axes[i] = static_cast<int64_t>(i);
    }
    if (inputs.size() > 3 && inputs[3].dtype != DataType::Unknown) {
        axes = ReadInt64s(inputs[3], "axes", context);
        if (axes.size() != count) {
            throw InferenceException(context + ": Slice axes must match starts in length");
        }
    }
    std::vector<int64_t> steps(count, 1);
    if (inputs.size() > 4 && inputs[4].dtype != DataType::Unknown) {
        steps = ReadInt64s(inputs[4], "steps", context);
        if (steps.size() != count) {
            throw InferenceException(context + ": Slice steps must match starts in length");
        }
    }

    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (static_cast<std::size_t>(rank) > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the slice limit of 16");
    }
    std::vector<int64_t> begins(static_cast<std::size_t>(rank), 0);
    std::vector<int64_t> strides(static_cast<std::size_t>(rank), 1);
    std::vector<int64_t> expected = in.dims;
    for (std::size_t i = 0; i < count; ++i) {
        const int64_t step = steps[i];
        if (step == 0) {
            throw InferenceException(context + ": Slice steps must be non-zero");
        }
        int64_t axis = axes[i];
        if (axis < -rank || axis >= rank) {
            throw InferenceException(context + ": Slice axis is out of range");
        }
        if (axis < 0) {
            axis += rank;
        }
        const std::size_t ua = static_cast<std::size_t>(axis);
        const int64_t dim = in.dims[ua];
        int64_t begin = starts[i];
        int64_t end = ends[i];
        if (begin < 0) {
            begin += dim;
        }
        if (end < 0) {
            end += dim;
        }
        if (step > 0) {
            begin = std::max<int64_t>(0, std::min(begin, dim));
            end = std::max<int64_t>(0, std::min(end, dim));
            expected[ua] = end > begin ? (end - begin + step - 1) / step : 0;
        } else {
            begin = std::max<int64_t>(0, std::min(begin, dim - 1));
            end = std::max<int64_t>(-1, std::min(end, dim - 1));
            expected[ua] = dim > 0 && begin > end ? (begin - end - step - 1) / -step : 0;
        }
        begins[ua] = begin;
        strides[ua] = step;
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the sliced ranges");
    }

    // Empty selections carry no elements; only the shape contract applies.
    int64_t out_count = 1;
    for (const int64_t d : out.dims) {
        if (d < 0) {
            throw InferenceException(context + ": non-positive dimension");
        }
        out_count *= d;
    }
    (void)ElementCount(in.dims, context);
    if (out_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    if (out_count == 0) {
        return;
    }

    const IndexSet beginsSet = Pack(begins, context);
    const IndexSet stepsSet = Pack(strides, context);
    const IndexSet inStrides = Pack(Strides(in.dims), context);
    const IndexSet outStrides = Pack(Strides(out.dims), context);
    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    if (in.dtype == DataType::Float32) {
        Launch(opencl_runtime::KernelFor("SliceF", kSourceFloat, "slice_f"), x, y, beginsSet, stepsSet, inStrides,
               outStrides, out_count, static_cast<int>(rank), context);
    } else if (in.dtype == DataType::Int64) {
        Launch(opencl_runtime::KernelFor("SliceI", kSourceInt, "slice_i"), x, y, beginsSet, stepsSet, inStrides,
               outStrides, out_count, static_cast<int>(rank), context);
    } else {
        Launch(opencl_runtime::KernelFor("SliceB", kSourceBool, "slice_b"), x, y, beginsSet, stepsSet, inStrides,
               outStrides, out_count, static_cast<int>(rank), context);
    }
}

} // namespace opencl_kernels
} // namespace nert
