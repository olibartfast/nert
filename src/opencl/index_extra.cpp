// Index and shape rows on OpenCL: Identity, Squeeze, CumSum, EyeLike,
// ScatterND, and ReduceMean.
//
// Mirrors the CPU oracles: Identity and Squeeze copy the input buffer byte
// for byte (shape inference owns the output dims) and take any rank; CumSum
// scans one axis lane per work-item (float32 accumulates in double and rounds
// once per output element, int64 accumulates exactly; `exclusive`/`reverse`
// default to 0 and the axis is an int64 scalar input read back to the host);
// EyeLike paints ones on the k-th diagonal and zeros elsewhere; ScatterND
// copies `data` and then applies the `updates` slices with one kernel launch
// each in slice order (the int64 indices are read back to the host, which
// validates them against the oracle, wraps negatives, and computes each
// slice's base offset, so duplicate indices apply in order and the last
// duplicate wins); ReduceMean averages over the requested axes in double (the
// device must report cl_khr_fp64; the axes resolution matches the oracle, so
// an absent axes input reduces every axis and noop_with_empty_axes applies
// only to a present-but-empty axes input). CumSum, ScatterND, and ReduceMean
// cap rank at 16 where index math is packed into kernel arguments. The OpenCL
// C sources compile once per process via the shared runtime.

#include "nert/device.hpp"
#include "runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
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

std::size_t ElementSize(DataType dtype, const std::string& context) {
    switch (dtype) {
    case DataType::Float32:
        return sizeof(float);
    case DataType::Int64:
        return sizeof(int64_t);
    case DataType::Bool:
        return sizeof(bool);
    default:
        throw InferenceException(context + ": tensors must be float32, int64, or bool");
    }
}

std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

// Reads an int64 tensor back to the host. The rows in this file validate
// values (the CumSum axis, the ScatterND indices, the ReduceMean axes)
// exactly like the CPU oracles, which need the numbers before any launch.
std::vector<int64_t> ReadInt64(const TensorView& view, int64_t count, const std::string& context) {
    std::vector<int64_t> host(static_cast<std::size_t>(count), 0);
    cl_mem mem = reinterpret_cast<cl_mem>(const_cast<void*>(view.data));
    CheckCl(clEnqueueReadBuffer(opencl_runtime::Queue(), mem, CL_TRUE, 0,
                                static_cast<std::size_t>(count) * sizeof(int64_t), host.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer", context);
    return host;
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

constexpr const char* kCumSumSource = R"CLC(
__kernel void cum_sum_f32(__global const float* x, __global float* y, long axisDim, long inner, long total,
                          int exclusive, int reverse) {
    long o = (long)get_global_id(0);
    if (o >= total) { return; }
    long n = o % inner;
    long lane = o / inner;
    long i = lane % axisDim;
    long base = lane - i;
    double acc = 0.0;
    if (reverse == 0) {
        const long end = (exclusive != 0) ? i - 1 : i;
        for (long p = 0; p <= end; ++p) { acc += (double)x[(base + p) * inner + n]; }
    } else {
        const long end = (exclusive != 0) ? i + 1 : i;
        for (long p = axisDim - 1; p >= end; --p) { acc += (double)x[(base + p) * inner + n]; }
    }
    y[o] = (float)acc;
}
__kernel void cum_sum_i64(__global const long* x, __global long* y, long axisDim, long inner, long total,
                          int exclusive, int reverse) {
    long o = (long)get_global_id(0);
    if (o >= total) { return; }
    long n = o % inner;
    long lane = o / inner;
    long i = lane % axisDim;
    long base = lane - i;
    long acc = 0;
    if (reverse == 0) {
        const long end = (exclusive != 0) ? i - 1 : i;
        for (long p = 0; p <= end; ++p) { acc += x[(base + p) * inner + n]; }
    } else {
        const long end = (exclusive != 0) ? i + 1 : i;
        for (long p = axisDim - 1; p >= end; --p) { acc += x[(base + p) * inner + n]; }
    }
    y[o] = acc;
}
)CLC";

constexpr const char* kEyeLikeSource = R"CLC(
__kernel void eye_like_f(__global float* y, long cols, long k, long total) {
    long i = (long)get_global_id(0);
    if (i >= total) { return; }
    long r = i / cols;
    long c = i - r * cols;
    y[i] = (c - r == k) ? 1.0f : 0.0f;
}
__kernel void eye_like_i64(__global long* y, long cols, long k, long total) {
    long i = (long)get_global_id(0);
    if (i >= total) { return; }
    long r = i / cols;
    long c = i - r * cols;
    y[i] = (c - r == k) ? 1L : 0L;
}
__kernel void eye_like_b(__global uchar* y, long cols, long k, long total) {
    long i = (long)get_global_id(0);
    if (i >= total) { return; }
    long r = i / cols;
    long c = i - r * cols;
    y[i] = (c - r == k) ? (uchar)1 : (uchar)0;
}
)CLC";

// One launch per update slice: the host computes each slice's target base
// offset from the indices read back before the launch, and the in-order
// queue applies the slices in slice order, so duplicate indices apply in
// order and the last duplicate wins, exactly like the CPU oracle.
constexpr const char* kScatterNDSliceSource = R"CLC(
#define SCATTER_SLICE_BODY \
    long e = (long)get_global_id(0); \
    if (e >= sliceSize) { return; } \
    out[base + e] = upd[uOff + e];
__kernel void scatter_slice_f32(__global const float* upd, __global float* out, long uOff, long base,
                                long sliceSize) {
    SCATTER_SLICE_BODY
}
__kernel void scatter_slice_i64(__global const long* upd, __global long* out, long uOff, long base,
                                long sliceSize) {
    SCATTER_SLICE_BODY
}
__kernel void scatter_slice_b(__global const uchar* upd, __global uchar* out, long uOff, long base, long sliceSize) {
    SCATTER_SLICE_BODY
}
)CLC";

// Everything the mean kernel needs to map one output element onto its
// reduced input span. Passed by value as one kernel argument; every member
// is an 8-byte integer, so the host and device layouts agree.
struct ReduceParams {
    cl_long rank;
    cl_long numReduced;
    cl_long reducedCount;
    cl_long outCount;
    cl_long groupSize;
    cl_long outStride[kMaxRank];
    cl_long inStride[kMaxRank];
    cl_long outAxis[kMaxRank]; // per input axis: output axis, or -1 when reduced
    cl_long redDims[kMaxRank];
    cl_long redStride[kMaxRank];
    cl_long redInStride[kMaxRank];
};
static_assert(sizeof(ReduceParams) == 5 * sizeof(cl_long) + 6 * kMaxRank * sizeof(cl_long),
              "device struct layout must match");

constexpr const char* kMeanSource = R"CLC(
typedef struct {
    long rank;
    long numReduced;
    long reducedCount;
    long outCount;
    long groupSize;
    long outStride[16];
    long inStride[16];
    long outAxis[16];
    long redDims[16];
    long redStride[16];
    long redInStride[16];
} ReduceParams;

// One work-item per output element: walk the surviving coordinates for the
// base input offset, then gather the reduced span in row-major order,
// accumulating in double and dividing once by the group size.
__kernel void reduce_mean(__global const float* x, __global float* y, ReduceParams p) {
    long o = (long)get_global_id(0);
    if (o < p.outCount) {
        long base = 0;
        long rem = o;
        for (long j = 0; j < p.rank; ++j) {
            long a = p.outAxis[j];
            if (a >= 0) {
                long coord = rem / p.outStride[a];
                rem -= coord * p.outStride[a];
                base += coord * p.inStride[j];
            }
        }
        double acc = 0.0;
        for (long r = 0; r < p.reducedCount; ++r) {
            long off = base;
            long t = r;
            for (long j = 0; j < p.numReduced; ++j) {
                long coord = t / p.redStride[j];
                t -= coord * p.redStride[j];
                off += coord * p.redInStride[j];
            }
            acc += (double)x[off];
        }
        y[o] = (float)(acc / (double)p.groupSize);
    }
}
)CLC";

void CopyDeviceBuffer(const TensorView& src, const TensorView& dst, std::size_t bytes, const std::string& context) {
    cl_mem sm = reinterpret_cast<cl_mem>(const_cast<void*>(src.data));
    cl_mem dm = reinterpret_cast<cl_mem>(dst.data);
    CheckCl(clEnqueueCopyBuffer(opencl_runtime::Queue(), sm, dm, 0, 0, bytes, 0, nullptr, nullptr),
            "clEnqueueCopyBuffer", context);
}

} // namespace

void Identity(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Identity (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != out.dtype) {
        throw InferenceException(context + ": input and output dtypes differ");
    }
    const std::size_t element = ElementSize(in.dtype, context);
    if (in.dims != out.dims) {
        throw InferenceException(context + ": input and output shapes differ");
    }
    const int64_t count = ElementCount(out.dims, context);
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    const std::size_t bytes = static_cast<std::size_t>(count) * element;
    if (bytes == 0) {
        return;
    }
    if (in.data != out.data) {
        CopyDeviceBuffer(in, out, bytes, context);
    }
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

void Squeeze(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Squeeze (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected a data input, an optional axes input, and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != out.dtype) {
        throw InferenceException(context + ": input and output dtypes differ");
    }
    const std::size_t element = ElementSize(in.dtype, context);
    const int64_t count = ElementCount(out.dims, context);
    if (ElementCount(in.dims, context) != count) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    const std::size_t bytes = static_cast<std::size_t>(count) * element;
    if (bytes == 0) {
        return;
    }
    if (in.data != out.data) {
        CopyDeviceBuffer(in, out, bytes, context);
    }
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

void CumSum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "CumSum (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected a data input, an axis input, and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& axisView = inputs[1];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": input must be float32 or int64");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output must share the input dtype");
    }
    if (axisView.dtype != DataType::Int64) {
        throw InferenceException(context + ": axis input must be int64");
    }
    if (ElementCount(axisView.dims, context) != 1) {
        throw InferenceException(context + ": axis input must be a scalar");
    }
    if (axisView.data == nullptr) {
        throw InferenceException(context + ": null axis buffer");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (rank == 0) {
        throw InferenceException(context + ": input must have rank >= 1");
    }
    if (rank > static_cast<int64_t>(kMaxRank)) {
        throw InferenceException(context + ": rank exceeds the pilot limit of 16");
    }
    // The axis scalar lives in device memory; read it back so the range
    // check matches the CPU oracle before any launch.
    const int64_t axisValue = ReadInt64(axisView, 1, context)[0];
    if (axisValue < -rank || axisValue >= rank) {
        throw InferenceException(context + ": axis out of range");
    }
    const int64_t axis = axisValue < 0 ? axisValue + rank : axisValue;
    const int64_t exclusive = IntAttribute(node, "exclusive", 0, context);
    if (exclusive != 0 && exclusive != 1) {
        throw InferenceException(context + ": exclusive must be 0 or 1 here");
    }
    const int64_t reverse = IntAttribute(node, "reverse", 0, context);
    if (reverse != 0 && reverse != 1) {
        throw InferenceException(context + ": reverse must be 0 or 1 here");
    }
    if (out.dims != in.dims) {
        throw InferenceException(context + ": output shape must match the input");
    }
    const int64_t outCount = ElementCount(out.dims, context);
    if (outCount > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    if (outCount == 0) {
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
    const int64_t axisDim = in.dims[static_cast<std::size_t>(axis)];

    const bool is_float = in.dtype == DataType::Float32;
    cl_kernel kernel = is_float ? opencl_runtime::KernelFor("CumSum/float", kCumSumSource, "cum_sum_f32")
                                : opencl_runtime::KernelFor("CumSum/int64", kCumSumSource, "cum_sum_i64");
    cl_mem xm = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    const cl_long axis_dim_arg = axisDim;
    const cl_long inner_arg = inner;
    const cl_long total_arg = outCount;
    const cl_int exclusive_arg = static_cast<cl_int>(exclusive);
    const cl_int reverse_arg = static_cast<cl_int>(reverse);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(xm), &xm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(axis_dim_arg), &axis_dim_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(inner_arg), &inner_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(total_arg), &total_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(exclusive_arg), &exclusive_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(reverse_arg), &reverse_arg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(outCount);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

void EyeLike(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "EyeLike (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dims.size() != 2 || out.dims != in.dims) {
        throw InferenceException(context + ": input and output must share a rank-2 shape");
    }
    if (out.dtype != DataType::Float32 && out.dtype != DataType::Int64 && out.dtype != DataType::Bool) {
        throw InferenceException(context + ": output must be float32, int64, or bool");
    }
    const int64_t k = IntAttribute(node, "k", 0, context);
    const int64_t cols = in.dims[1];
    const int64_t count = ElementCount(out.dims, context);
    if (out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (count == 0) {
        return;
    }

    const char* entry = nullptr;
    const char* cache_key = nullptr;
    if (out.dtype == DataType::Float32) {
        entry = "eye_like_f";
        cache_key = "EyeLike/float";
    } else if (out.dtype == DataType::Int64) {
        entry = "eye_like_i64";
        cache_key = "EyeLike/int64";
    } else {
        entry = "eye_like_b";
        cache_key = "EyeLike/bool";
    }
    cl_kernel kernel = opencl_runtime::KernelFor(cache_key, kEyeLikeSource, entry);
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    const cl_long cols_arg = cols;
    const cl_long k_arg = k;
    const cl_long total_arg = count;
    CheckCl(clSetKernelArg(kernel, 0, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(cols_arg), &cols_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(k_arg), &k_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(total_arg), &total_arg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

void ScatterND(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ScatterND (node '" + node.name + "')";
    if (inputs.size() != 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected data, indices, updates, and one output");
    }
    // Only reduction "none" is supported. The CPU oracle has no reduction
    // option at all (it always replaces the targeted slice and ignores
    // attributes), so this check is deliberately stricter than the oracle:
    // a pilot-side guard that rejects unsupported attribute values up
    // front, before any device work.
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "reduction") {
            const std::string* value = std::get_if<std::string>(&attr.value);
            if (value == nullptr || *value != "none") {
                throw InferenceException(context + ": only reduction 'none' is supported");
            }
        }
    }
    const TensorView& data = inputs[0];
    const TensorView& indices = inputs[1];
    const TensorView& updates = inputs[2];
    const TensorView& out = outputs[0];
    if (data.dtype != DataType::Float32 && data.dtype != DataType::Int64 && data.dtype != DataType::Bool) {
        throw InferenceException(context + ": data must be float32, int64, or bool");
    }
    if (updates.dtype != data.dtype || out.dtype != data.dtype) {
        throw InferenceException(context + ": data, updates, and output must share dtype");
    }
    if (indices.dtype != DataType::Int64) {
        throw InferenceException(context + ": indices must be int64");
    }
    if (indices.dims.empty()) {
        throw InferenceException(context + ": indices must have rank >= 1");
    }
    const int64_t dataRank = static_cast<int64_t>(data.dims.size());
    if (dataRank > static_cast<int64_t>(kMaxRank)) {
        throw InferenceException(context + ": rank exceeds the pilot limit of 16");
    }
    const int64_t k = indices.dims.back();
    if (k < 1 || k > dataRank) {
        throw InferenceException(context + ": indices last dim must be within the data rank");
    }
    if (out.dims != data.dims) {
        throw InferenceException(context + ": output shape must match the data shape");
    }
    std::vector<int64_t> expectedUpdates(indices.dims.begin(), indices.dims.end() - 1);
    for (int64_t i = k; i < dataRank; ++i) {
        expectedUpdates.push_back(data.dims[static_cast<std::size_t>(i)]);
    }
    if (updates.dims != expectedUpdates) {
        throw InferenceException(context + ": updates shape does not match indices and data");
    }
    const int64_t dataCount = ElementCount(data.dims, context);
    const int64_t indicesCount = ElementCount(indices.dims, context);
    (void)ElementCount(updates.dims, context);
    if (data.data == nullptr || indices.data == nullptr || updates.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    // The indices live on the device; read them back so the wrap and range
    // checks match the CPU oracle before the scatter launches.
    const std::vector<int64_t> idx = ReadInt64(indices, indicesCount, context);
    const std::vector<int64_t> dataStrides = Strides(data.dims);
    int64_t sliceCount = 1;
    for (std::size_t i = 0; i + 1 < indices.dims.size(); ++i) {
        sliceCount *= indices.dims[i];
    }
    int64_t sliceSize = 1;
    for (int64_t i = k; i < dataRank; ++i) {
        sliceSize *= data.dims[static_cast<std::size_t>(i)];
    }
    for (int64_t s = 0; s < sliceCount; ++s) {
        for (int64_t j = 0; j < k; ++j) {
            const int64_t coord = idx[static_cast<std::size_t>(s * k + j)];
            const int64_t dim = data.dims[static_cast<std::size_t>(j)];
            if (coord < -dim || coord >= dim) {
                throw InferenceException(context + ": index out of range");
            }
        }
    }

    const std::size_t elemSize = ElementSize(data.dtype, context);
    const std::size_t bytes = static_cast<std::size_t>(dataCount) * elemSize;
    if (bytes > 0) {
        // Copy data first; the in-order queue runs the scatter launches
        // after the copy completes.
        CopyDeviceBuffer(data, out, bytes, context);
    }
    const int64_t total = sliceCount * sliceSize;
    if (total > 0) {
        const char* entry = nullptr;
        const char* cache_key = nullptr;
        if (data.dtype == DataType::Float32) {
            entry = "scatter_slice_f32";
            cache_key = "ScatterND/float";
        } else if (data.dtype == DataType::Int64) {
            entry = "scatter_slice_i64";
            cache_key = "ScatterND/int64";
        } else {
            entry = "scatter_slice_b";
            cache_key = "ScatterND/bool";
        }
        cl_kernel kernel = opencl_runtime::KernelFor(cache_key, kScatterNDSliceSource, entry);
        cl_mem updm = reinterpret_cast<cl_mem>(const_cast<void*>(updates.data));
        cl_mem outm = reinterpret_cast<cl_mem>(out.data);
        const cl_long slice_size_arg = sliceSize;
        CheckCl(clSetKernelArg(kernel, 0, sizeof(updm), &updm), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 1, sizeof(outm), &outm), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 4, sizeof(slice_size_arg), &slice_size_arg), "clSetKernelArg", context);
        // One launch per update slice, in slice order: the in-order queue
        // serializes the launches, so a duplicated target offset is
        // overwritten by the later slice and the last duplicate wins,
        // exactly like the CPU oracle. The base offsets are computed here
        // from the indices read back above; the device does no index math.
        const std::size_t global = static_cast<std::size_t>(sliceSize);
        for (int64_t s = 0; s < sliceCount; ++s) {
            int64_t base = 0;
            for (int64_t j = 0; j < k; ++j) {
                int64_t coord = idx[static_cast<std::size_t>(s * k + j)];
                if (coord < 0) {
                    coord += data.dims[static_cast<std::size_t>(j)];
                }
                base += coord * dataStrides[static_cast<std::size_t>(j)];
            }
            const cl_long u_off_arg = s * sliceSize;
            const cl_long base_arg = base;
            CheckCl(clSetKernelArg(kernel, 2, sizeof(u_off_arg), &u_off_arg), "clSetKernelArg", context);
            CheckCl(clSetKernelArg(kernel, 3, sizeof(base_arg), &base_arg), "clSetKernelArg", context);
            CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr,
                                           nullptr),
                    "clEnqueueNDRangeKernel", context);
        }
    }
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

void ReduceMean(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ReduceMean (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one or two inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": data tensors must be float32");
    }
    const std::vector<int64_t>& inDims = in.dims;
    const int64_t rank = static_cast<int64_t>(inDims.size());
    if (rank > static_cast<int64_t>(kMaxRank)) {
        throw InferenceException(context + ": rank exceeds the pilot limit of 16");
    }
    const int64_t inCount = ElementCount(inDims, context);
    if (in.data == nullptr) {
        throw InferenceException(context + ": null input buffer");
    }

    const int64_t keepdims = IntAttribute(node, "keepdims", 1, context);
    const int64_t noopEmpty = IntAttribute(node, "noop_with_empty_axes", 0, context);

    // Resolve the axes exactly like the CPU oracle (src/kernels/
    // reduce_mean.cpp): an absent axes input reduces every axis and
    // noop_with_empty_axes does not apply there; a present axes input
    // carries the axes, and only a present-but-empty axes input turns the
    // attribute into an identity copy. (A zero-element axes tensor is
    // rejected by the dimension check today, so the identity branch mirrors
    // the oracle without currently firing.)
    std::vector<int64_t> axes;
    bool identity = false;
    if (inputs.size() == 2) {
        const TensorView& axesView = inputs[1];
        if (axesView.dtype != DataType::Int64) {
            throw InferenceException(context + ": axes tensor must be int64");
        }
        const int64_t axesCount = ElementCount(axesView.dims, context);
        if (axesCount > 0) {
            if (axesView.data == nullptr) {
                throw InferenceException(context + ": null axes buffer");
            }
            // The axes tensor lives in device memory; read the small values
            // back so the host can validate them like the CPU oracle.
            axes = ReadInt64(axesView, axesCount, context);
        } else if (noopEmpty != 0) {
            identity = true;
        } else {
            for (int64_t i = 0; i < rank; ++i) {
                axes.push_back(i);
            }
        }
    } else {
        // The axes input is absent: reduce every axis; noop_with_empty_axes
        // does not apply (the CPU oracle ignores it here too).
        for (int64_t i = 0; i < rank; ++i) {
            axes.push_back(i);
        }
    }

    if (out.data == nullptr) {
        throw InferenceException(context + ": null output buffer");
    }

    if (identity) {
        if (out.dims != inDims) {
            throw InferenceException(context + ": output shape does not match the input");
        }
        (void)ElementCount(out.dims, context);
        const std::size_t bytes = static_cast<std::size_t>(inCount) * sizeof(float);
        if (bytes > 0 && in.data != out.data) {
            CopyDeviceBuffer(in, out, bytes, context);
        }
        CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
        return;
    }

    // Normalize and validate the axes: unique and within [-rank, rank - 1].
    std::vector<bool> reduced(static_cast<std::size_t>(rank), false);
    for (int64_t axis : axes) {
        if (axis < -rank || axis >= rank) {
            throw InferenceException(context + ": axis out of range");
        }
        const int64_t normalized = axis < 0 ? axis + rank : axis;
        if (reduced[static_cast<std::size_t>(normalized)]) {
            throw InferenceException(context + ": duplicate axis");
        }
        reduced[static_cast<std::size_t>(normalized)] = true;
    }

    std::vector<int64_t> expectedDims;
    for (int64_t i = 0; i < rank; ++i) {
        if (reduced[static_cast<std::size_t>(i)]) {
            if (keepdims != 0) {
                expectedDims.push_back(1);
            }
        } else {
            expectedDims.push_back(inDims[static_cast<std::size_t>(i)]);
        }
    }
    if (out.dims != expectedDims) {
        throw InferenceException(context + ": output shape does not match the reduced axes");
    }
    const int64_t outCount = ElementCount(out.dims, context);

    // Contiguous row-major strides for the input and the output, the output
    // axis per input axis, and the reduced span per output element.
    std::vector<int64_t> inStride(inDims.size(), 1);
    for (std::size_t i = inDims.size(); i-- > 1;) {
        inStride[i - 1] = inStride[i] * inDims[i];
    }
    std::vector<int64_t> outStride(expectedDims.size(), 1);
    for (std::size_t i = expectedDims.size(); i-- > 1;) {
        outStride[i - 1] = outStride[i] * expectedDims[i];
    }
    std::vector<int64_t> outAxis(inDims.size(), -1);
    int64_t nextOutAxis = 0;
    for (int64_t i = 0; i < rank; ++i) {
        if (!reduced[static_cast<std::size_t>(i)]) {
            outAxis[static_cast<std::size_t>(i)] = keepdims != 0 ? i : nextOutAxis;
            ++nextOutAxis;
        }
    }
    std::vector<int64_t> redAxes;
    for (int64_t i = 0; i < rank; ++i) {
        if (reduced[static_cast<std::size_t>(i)]) {
            redAxes.push_back(i);
        }
    }
    std::vector<int64_t> redDims(redAxes.size(), 0);
    std::vector<int64_t> redStride(redAxes.size(), 1);
    std::vector<int64_t> redInStride(redAxes.size(), 1);
    int64_t reducedCount = 1;
    for (std::size_t j = redAxes.size(); j-- > 0;) {
        redStride[j] = reducedCount;
        redDims[j] = inDims[static_cast<std::size_t>(redAxes[j])];
        redInStride[j] = inStride[static_cast<std::size_t>(redAxes[j])];
        reducedCount *= redDims[j];
    }

    ReduceParams params{};
    params.rank = rank;
    params.numReduced = static_cast<cl_long>(redDims.size());
    params.reducedCount = static_cast<cl_long>(reducedCount);
    params.outCount = static_cast<cl_long>(outCount);
    params.groupSize = static_cast<cl_long>(inCount / outCount);
    for (std::size_t i = 0; i < outStride.size() && i < kMaxRank; ++i) {
        params.outStride[i] = outStride[i];
    }
    for (std::size_t i = 0; i < inStride.size() && i < kMaxRank; ++i) {
        params.inStride[i] = inStride[i];
    }
    for (std::size_t i = 0; i < outAxis.size() && i < kMaxRank; ++i) {
        params.outAxis[i] = outAxis[i];
    }
    for (std::size_t i = 0; i < redDims.size() && i < kMaxRank; ++i) {
        params.redDims[i] = redDims[i];
        params.redStride[i] = redStride[i];
        params.redInStride[i] = redInStride[i];
    }

    // The float32 kernel accumulates in double; OpenCL fp64 is optional, so
    // fail with the device capability named instead of a build error.
    cl_device_id device = nullptr;
    CheckCl(clGetContextInfo(opencl_runtime::Context(), CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr),
            "clGetContextInfo", context);
    cl_device_fp_config fp64 = 0;
    CheckCl(clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp64), &fp64, nullptr), "clGetDeviceInfo",
            context);
    if (fp64 == 0) {
        throw InferenceException(context + ": device has no double-precision support (cl_khr_fp64 is required)");
    }

    cl_kernel kernel = opencl_runtime::KernelFor("ReduceMean", kMeanSource, "reduce_mean");
    cl_mem xm = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(xm), &xm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(params), &params), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(outCount);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
