// Reduction rows on OpenCL: ReduceL2 and ArgMax.
//
// ReduceL2 mirrors the CPU oracle's axes/keepdims/noop handling and one
// work-item per output, walking the reduced axes; ArgMax walks one axis and
// returns int64 indices. Accumulation is float32, compared with tolerance.

#include "../reduce_axes.hpp"
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

struct StrideSet {
    cl_long s[kMaxRank];
};

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

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback, const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute '" + name + "' must be an integer");
            }
            return *value;
        }
    }
    return fallback;
}

constexpr const char* kSource = R"CLC(
typedef struct { long s[16]; } StrideSet;
__kernel void reduce_l2(__global const float* x, __global float* y, long outCount, StrideSet outStrides,
                        StrideSet outInStride, StrideSet redSizes, StrideSet redStrides, int redCount) {
    long i = get_global_id(0);
    if (i >= outCount) {
        return;
    }
    long remaining = i;
    long base = 0;
    for (int j = 0; j < 16; ++j) {
        if (outStrides.s[j] == 0) {
            continue;
        }
        long coord = remaining / outStrides.s[j];
        remaining %= outStrides.s[j];
        base += coord * outInStride.s[j];
    }
    long total = 1;
    for (int j = 0; j < redCount; ++j) {
        total *= redSizes.s[j];
    }
    float acc = 0.0f;
    for (long r = 0; r < total; ++r) {
        long rem = r;
        long offset = base;
        for (int j = redCount - 1; j >= 0; --j) {
            long coord = rem % redSizes.s[j];
            rem /= redSizes.s[j];
            offset += coord * redStrides.s[j];
        }
        float v = x[offset];
        acc += v * v;
    }
    y[i] = sqrt(acc);
}
__kernel void arg_max_f(__global const float* x, __global long* y, long outer, long dim, long inner, int last) {
    long idx = get_global_id(0);
    long o = idx / inner;
    long n = idx % inner;
    long base = o * dim * inner + n;
    float bestVal = x[base];
    long best = 0;
    for (long i = 1; i < dim; ++i) {
        float v = x[base + i * inner];
        if (v > bestVal || (last != 0 && v == bestVal)) {
            bestVal = v;
            best = i;
        }
    }
    y[idx] = best;
}
__kernel void arg_max_i(__global const long* x, __global long* y, long outer, long dim, long inner, int last) {
    long idx = get_global_id(0);
    long o = idx / inner;
    long n = idx % inner;
    long base = o * dim * inner + n;
    long bestVal = x[base];
    long best = 0;
    for (long i = 1; i < dim; ++i) {
        long v = x[base + i * inner];
        if (v > bestVal || (last != 0 && v == bestVal)) {
            bestVal = v;
            best = i;
        }
    }
    y[idx] = best;
}
)CLC";

} // namespace

void ReduceL2(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ReduceL2 (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one or two inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": data tensors must be float32");
    }
    const std::vector<int64_t>& in_dims = in.dims;
    const int64_t rank = static_cast<int64_t>(in_dims.size());
    const int64_t in_count = ElementCount(in_dims, context);
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (rank > static_cast<int64_t>(kMaxRank)) {
        throw InferenceException(context + ": rank exceeds the OpenCL reduction limit of 16");
    }
    const bool keepdims = IntAttribute(node, "keepdims", 1, context) != 0;
    const ReduceAxes resolved_axes = ResolveReduceAxes(node);

    // Resolve the axes once: the old attribute form reduces all axes when the
    // attribute is absent or empty; the input form treats an absent input as
    // all axes, and only a present empty input follows noop_with_empty_axes.
    std::vector<int64_t> axes;
    bool identity = false;
    if (resolved_axes.form == ReduceAxesForm::Attribute) {
        if (resolved_axes.axes_attribute != nullptr) {
            const std::vector<int64_t>* values = ReduceAttributeIntList(resolved_axes.axes_attribute);
            if (values == nullptr) {
                throw InferenceException(context + ": attribute 'axes' must be an integer list");
            }
            axes = *values;
        }
        if (axes.empty()) {
            for (int64_t i = 0; i < rank; ++i) {
                axes.push_back(i);
            }
        }
    } else {
        bool axes_supplied = false;
        if (inputs.size() == 2 && inputs[1].dtype != DataType::Unknown) {
            axes_supplied = true;
            const TensorView& axes_view = inputs[1];
            if (axes_view.dtype != DataType::Int64) {
                throw InferenceException(context + ": axes tensor must be int64");
            }
            const int64_t axes_count = ReduceAxesElementCount(axes_view.dims, context);
            if (axes_count > 0) {
                if (axes_view.data == nullptr) {
                    throw InferenceException(context + ": null axes buffer");
                }
                axes.resize(static_cast<std::size_t>(axes_count), 0);
                CheckCl(clEnqueueReadBuffer(opencl_runtime::Queue(),
                                            reinterpret_cast<cl_mem>(const_cast<void*>(axes_view.data)), CL_TRUE, 0,
                                            static_cast<std::size_t>(axes_count) * sizeof(int64_t), axes.data(), 0,
                                            nullptr, nullptr),
                        "clEnqueueReadBuffer", context);
            }
        }
        if (axes.empty()) {
            if (axes_supplied && resolved_axes.noop_with_empty_axes) {
                identity = true;
            } else {
                for (int64_t i = 0; i < rank; ++i) {
                    axes.push_back(i);
                }
            }
        }
    }
    if (identity) {
        if (out.dims != in_dims) {
            throw InferenceException(context + ": output shape does not match the input");
        }
        CheckCl(clEnqueueCopyBuffer(opencl_runtime::Queue(), reinterpret_cast<cl_mem>(const_cast<void*>(in.data)),
                                    reinterpret_cast<cl_mem>(out.data), 0, 0,
                                    static_cast<std::size_t>(in_count) * sizeof(float), 0, nullptr, nullptr),
                "clEnqueueCopyBuffer", context);
        CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
        return;
    }

    std::vector<bool> reduced(static_cast<std::size_t>(rank), false);
    for (const int64_t axis : axes) {
        if (axis < -rank || axis >= rank) {
            throw InferenceException(context + ": axis out of range");
        }
        const int64_t normalized = axis < 0 ? axis + rank : axis;
        if (reduced[static_cast<std::size_t>(normalized)]) {
            throw InferenceException(context + ": duplicate axis");
        }
        reduced[static_cast<std::size_t>(normalized)] = true;
    }
    std::vector<int64_t> expected;
    for (int64_t i = 0; i < rank; ++i) {
        if (reduced[static_cast<std::size_t>(i)]) {
            if (keepdims) {
                expected.push_back(1);
            }
        } else {
            expected.push_back(in_dims[static_cast<std::size_t>(i)]);
        }
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the reduced axes");
    }
    const int64_t out_count = ElementCount(out.dims, context);

    const std::vector<int64_t> in_strides = Strides(in_dims);
    const std::vector<int64_t> out_strides = Strides(expected);
    std::vector<int64_t> out_in_stride(static_cast<std::size_t>(rank), 0);
    std::vector<int64_t> red_sizes;
    std::vector<int64_t> red_strides;
    if (keepdims) {
        for (int64_t i = 0; i < rank; ++i) {
            if (reduced[static_cast<std::size_t>(i)]) {
                red_sizes.push_back(in_dims[static_cast<std::size_t>(i)]);
                red_strides.push_back(in_strides[static_cast<std::size_t>(i)]);
            } else {
                out_in_stride[static_cast<std::size_t>(i)] = in_strides[static_cast<std::size_t>(i)];
            }
        }
    } else {
        int64_t out_axis = 0;
        for (int64_t i = 0; i < rank; ++i) {
            if (reduced[static_cast<std::size_t>(i)]) {
                red_sizes.push_back(in_dims[static_cast<std::size_t>(i)]);
                red_strides.push_back(in_strides[static_cast<std::size_t>(i)]);
            } else {
                out_in_stride[static_cast<std::size_t>(out_axis)] = in_strides[static_cast<std::size_t>(i)];
                ++out_axis;
            }
        }
    }
    if (red_sizes.size() > kMaxRank) {
        throw InferenceException(context + ": reduced rank exceeds the OpenCL limit of 16");
    }
    StrideSet out_strides_set{};
    StrideSet out_in_set{};
    StrideSet red_sizes_set{};
    StrideSet red_strides_set{};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        out_strides_set.s[i] = out_strides[i];
        out_in_set.s[i] = out_in_stride[i];
    }
    for (std::size_t i = 0; i < red_sizes.size(); ++i) {
        red_sizes_set.s[i] = red_sizes[i];
        red_strides_set.s[i] = red_strides[i];
    }
    const cl_int red_count = static_cast<cl_int>(red_sizes.size());
    const cl_long out_count_arg = out_count;
    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    cl_kernel kernel = opencl_runtime::KernelFor("ReduceL2", kSource, "reduce_l2");
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(out_count_arg), &out_count_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(out_strides_set), &out_strides_set), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(out_in_set), &out_in_set), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(red_sizes_set), &red_sizes_set), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(red_strides_set), &red_strides_set), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 7, sizeof(red_count), &red_count), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(out_count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

void ArgMax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ArgMax (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": input must be float32 or int64");
    }
    if (out.dtype != DataType::Int64) {
        throw InferenceException(context + ": output must be int64");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (rank == 0) {
        throw InferenceException(context + ": input must have rank >= 1");
    }
    int64_t axis = IntAttribute(node, "axis", 0, context);
    if (axis < -rank || axis >= rank) {
        throw InferenceException(context + ": axis out of range");
    }
    if (axis < 0) {
        axis += rank;
    }
    const int64_t keepdims = IntAttribute(node, "keepdims", 1, context);
    if (keepdims != 0 && keepdims != 1) {
        throw InferenceException(context + ": keepdims must be 0 or 1 here");
    }
    const int64_t last = IntAttribute(node, "select_last_index", 0, context);
    if (last != 0 && last != 1) {
        throw InferenceException(context + ": select_last_index must be 0 or 1 here");
    }
    const int64_t dim = in.dims[static_cast<std::size_t>(axis)];
    std::vector<int64_t> expected = in.dims;
    if (keepdims != 0) {
        expected[static_cast<std::size_t>(axis)] = 1;
    } else {
        expected.erase(expected.begin() + axis);
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the reduced axis");
    }
    (void)ElementCount(in.dims, context);
    const int64_t out_count = ElementCount(out.dims, context);
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
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= in.dims[static_cast<std::size_t>(i)];
    }
    const cl_long outer_arg = outer;
    const cl_long dim_arg = dim;
    const cl_long inner_arg = inner;
    const cl_int last_arg = static_cast<cl_int>(last);
    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    const char* entry = in.dtype == DataType::Float32 ? "arg_max_f" : "arg_max_i";
    const char* cache_key = in.dtype == DataType::Float32 ? "ArgMaxF" : "ArgMaxI";
    cl_kernel kernel = opencl_runtime::KernelFor(cache_key, kSource, entry);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(outer_arg), &outer_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(dim_arg), &dim_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(inner_arg), &inner_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(last_arg), &last_arg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(out_count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
