// More math rows on OpenCL: Clip, Log, Pow, Max, Min, Or.
//
// Mirrors the CPU oracles: Clip clamps float32 or int64 data with optional
// scalar bounds that live in device memory and are read back before the
// launch; Log is a float32 unary map with no special cases (negative -> NaN,
// 0 -> -inf); Pow broadcasts two float32 inputs and squares on an exact
// exponent of 2 like the oracle; Max/Min fold one shared float32 or int64
// dtype across NumPy-broadcast inputs in input order; Or folds bool inputs.
// One work-item per output element with zero strides on broadcast axes; the
// variadic rows carry up to eight pointers plus the input count. The OpenCL C
// source compiles once per process via the shared runtime, keyed per
// op/dtype variant.

#include "nert/device.hpp"
#include "runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

namespace nert {
namespace opencl_kernels {
namespace {

constexpr std::size_t kMaxRank = 16;
constexpr std::size_t kMaxInputs = 8;

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

// Per-input stride rows for the variadic fold kernels; rows past the real
// input count stay zero.
struct StrideTable {
    cl_long s[kMaxInputs][kMaxRank];
};

StrideTable PackTable(const std::vector<std::vector<int64_t>>& strides, const std::string& context) {
    if (strides.size() > kMaxInputs) {
        throw InferenceException(context + ": input count exceeds the pilot device limit of 8");
    }
    StrideTable packed{};
    for (std::size_t k = 0; k < strides.size(); ++k) {
        if (strides[k].size() > kMaxRank) {
            throw InferenceException(context + ": rank exceeds the pilot broadcast limit of 16");
        }
        for (std::size_t j = 0; j < strides[k].size(); ++j) {
            packed.s[k][j] = strides[k][j];
        }
    }
    return packed;
}

constexpr const char* kSource = R"CLC(
typedef struct { long s[16]; } StrideSet;
typedef struct { long s[8][16]; } StrideTable;
#define DECODE2 \
    long i = get_global_id(0); \
    if (i >= count) { return; } \
    long remaining = i, ao = 0, bo = 0; \
    for (int j = 0; j < rank; ++j) { \
        long coord = remaining / outStrides.s[j]; \
        remaining %= outStrides.s[j]; \
        ao += coord * aStrides.s[j]; \
        bo += coord * bStrides.s[j]; \
    }
#define DECODE8 \
    long i = get_global_id(0); \
    if (i >= count) { return; } \
    long remaining = i, o0 = 0, o1 = 0, o2 = 0, o3 = 0, o4 = 0, o5 = 0, o6 = 0, o7 = 0; \
    for (int j = 0; j < rank; ++j) { \
        long coord = remaining / outStrides.s[j]; \
        remaining %= outStrides.s[j]; \
        o0 += coord * inStrides.s[0][j]; \
        o1 += coord * inStrides.s[1][j]; \
        o2 += coord * inStrides.s[2][j]; \
        o3 += coord * inStrides.s[3][j]; \
        o4 += coord * inStrides.s[4][j]; \
        o5 += coord * inStrides.s[5][j]; \
        o6 += coord * inStrides.s[6][j]; \
        o7 += coord * inStrides.s[7][j]; \
    }
__kernel void clip_f(__global const float* x, __global float* y, long count, float lo, float hi) {
    long i = get_global_id(0);
    if (i < count) {
        float v = x[i] < lo ? lo : x[i];
        y[i] = hi < v ? hi : v;
    }
}
__kernel void clip_i(__global const long* x, __global long* y, long count, long lo, long hi) {
    long i = get_global_id(0);
    if (i < count) {
        long v = x[i] < lo ? lo : x[i];
        y[i] = hi < v ? hi : v;
    }
}
__kernel void log_f(__global const float* x, __global float* y, long count) {
    long i = get_global_id(0);
    if (i < count) {
        y[i] = log(x[i]);
    }
}
__kernel void pow_f(__global const float* a, __global const float* b, __global float* y, long count,
                    StrideSet outStrides, StrideSet aStrides, StrideSet bStrides, int rank) {
    DECODE2
    float base = a[ao];
    float exponent = b[bo];
    y[i] = exponent == 2.0f ? base * base : pow(base, exponent);
}
__kernel void max_f(__global const float* i0, __global const float* i1, __global const float* i2,
                    __global const float* i3, __global const float* i4, __global const float* i5,
                    __global const float* i6, __global const float* i7, int n, __global float* y, long count,
                    StrideSet outStrides, StrideTable inStrides, int rank) {
    DECODE8
    float best = i0[o0];
    if (1 < n) { float v = i1[o1]; best = best < v ? v : best; }
    if (2 < n) { float v = i2[o2]; best = best < v ? v : best; }
    if (3 < n) { float v = i3[o3]; best = best < v ? v : best; }
    if (4 < n) { float v = i4[o4]; best = best < v ? v : best; }
    if (5 < n) { float v = i5[o5]; best = best < v ? v : best; }
    if (6 < n) { float v = i6[o6]; best = best < v ? v : best; }
    if (7 < n) { float v = i7[o7]; best = best < v ? v : best; }
    y[i] = best;
}
__kernel void max_i(__global const long* i0, __global const long* i1, __global const long* i2,
                    __global const long* i3, __global const long* i4, __global const long* i5,
                    __global const long* i6, __global const long* i7, int n, __global long* y, long count,
                    StrideSet outStrides, StrideTable inStrides, int rank) {
    DECODE8
    long best = i0[o0];
    if (1 < n) { long v = i1[o1]; best = best < v ? v : best; }
    if (2 < n) { long v = i2[o2]; best = best < v ? v : best; }
    if (3 < n) { long v = i3[o3]; best = best < v ? v : best; }
    if (4 < n) { long v = i4[o4]; best = best < v ? v : best; }
    if (5 < n) { long v = i5[o5]; best = best < v ? v : best; }
    if (6 < n) { long v = i6[o6]; best = best < v ? v : best; }
    if (7 < n) { long v = i7[o7]; best = best < v ? v : best; }
    y[i] = best;
}
__kernel void min_f(__global const float* i0, __global const float* i1, __global const float* i2,
                    __global const float* i3, __global const float* i4, __global const float* i5,
                    __global const float* i6, __global const float* i7, int n, __global float* y, long count,
                    StrideSet outStrides, StrideTable inStrides, int rank) {
    DECODE8
    float best = i0[o0];
    if (1 < n) { float v = i1[o1]; best = v < best ? v : best; }
    if (2 < n) { float v = i2[o2]; best = v < best ? v : best; }
    if (3 < n) { float v = i3[o3]; best = v < best ? v : best; }
    if (4 < n) { float v = i4[o4]; best = v < best ? v : best; }
    if (5 < n) { float v = i5[o5]; best = v < best ? v : best; }
    if (6 < n) { float v = i6[o6]; best = v < best ? v : best; }
    if (7 < n) { float v = i7[o7]; best = v < best ? v : best; }
    y[i] = best;
}
__kernel void min_i(__global const long* i0, __global const long* i1, __global const long* i2,
                    __global const long* i3, __global const long* i4, __global const long* i5,
                    __global const long* i6, __global const long* i7, int n, __global long* y, long count,
                    StrideSet outStrides, StrideTable inStrides, int rank) {
    DECODE8
    long best = i0[o0];
    if (1 < n) { long v = i1[o1]; best = v < best ? v : best; }
    if (2 < n) { long v = i2[o2]; best = v < best ? v : best; }
    if (3 < n) { long v = i3[o3]; best = v < best ? v : best; }
    if (4 < n) { long v = i4[o4]; best = v < best ? v : best; }
    if (5 < n) { long v = i5[o5]; best = v < best ? v : best; }
    if (6 < n) { long v = i6[o6]; best = v < best ? v : best; }
    if (7 < n) { long v = i7[o7]; best = v < best ? v : best; }
    y[i] = best;
}
__kernel void or_b(__global const uchar* i0, __global const uchar* i1, __global const uchar* i2,
                   __global const uchar* i3, __global const uchar* i4, __global const uchar* i5,
                   __global const uchar* i6, __global const uchar* i7, int n, __global uchar* y, long count,
                   StrideSet outStrides, StrideTable inStrides, int rank) {
    DECODE8
    uchar acc = i0[o0];
    if (1 < n) { acc = (acc != 0 || i1[o1] != 0) ? (uchar)1 : (uchar)0; }
    if (2 < n) { acc = (acc != 0 || i2[o2] != 0) ? (uchar)1 : (uchar)0; }
    if (3 < n) { acc = (acc != 0 || i3[o3] != 0) ? (uchar)1 : (uchar)0; }
    if (4 < n) { acc = (acc != 0 || i4[o4] != 0) ? (uchar)1 : (uchar)0; }
    if (5 < n) { acc = (acc != 0 || i5[o5] != 0) ? (uchar)1 : (uchar)0; }
    if (6 < n) { acc = (acc != 0 || i6[o6] != 0) ? (uchar)1 : (uchar)0; }
    if (7 < n) { acc = (acc != 0 || i7[o7] != 0) ? (uchar)1 : (uchar)0; }
    y[i] = acc;
}
)CLC";

// Reads an optional scalar bound from device memory; returns `fallback` when
// the input is omitted (position past the end or an Unknown-dtype slot). A
// present bound must carry the data dtype, hold exactly one element, and have
// a live buffer; the single element is read back with a blocking
// clEnqueueReadBuffer before the kernel launches.
template <typename T>
T ReadBound(const std::vector<TensorView>& inputs, std::size_t index, T fallback, DataType dtype, const char* type_name,
            const std::string& context) {
    if (index >= inputs.size() || inputs[index].dtype == DataType::Unknown) {
        return fallback;
    }
    const TensorView& bound = inputs[index];
    if (bound.dtype != dtype) {
        throw InferenceException(context + ": min/max must be " + type_name);
    }
    if (ElementCount(bound.dims, context) != 1) {
        throw InferenceException(context + ": min/max must be scalar");
    }
    if (bound.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    cl_mem mem = reinterpret_cast<cl_mem>(bound.data);
    T value = fallback;
    CheckCl(clEnqueueReadBuffer(opencl_runtime::Queue(), mem, CL_TRUE, 0, sizeof(T), &value, 0, nullptr, nullptr),
            "clEnqueueReadBuffer", context);
    return value;
}

// Shared clamp launch for one dtype. Integer paths stay on integer
// comparisons (no float round-trip), and the absent-bound fallbacks make the
// clamp an exact identity; the ternaries mirror std::min(std::max(x, lo), hi)
// element for element, NaN included.
template <typename T>
void RunClip(const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs, const std::string& context,
             T lo_fallback, T hi_fallback, DataType dtype, const char* type_name, const char* entry,
             const char* cache_key) {
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    const int64_t count = ElementCount(out.dims, context);
    if (ElementCount(in.dims, context) != count) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    const T lo = ReadBound<T>(inputs, 1, lo_fallback, dtype, type_name, context);
    const T hi = ReadBound<T>(inputs, 2, hi_fallback, dtype, type_name, context);
    if (count == 0) {
        return;
    }
    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    const cl_long n = count;
    cl_kernel kernel = opencl_runtime::KernelFor(cache_key, kSource, entry);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(n), &n), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(lo), &lo), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(hi), &hi), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

// Shared launch for the variadic fold rows (Max, Min, Or): one work-item per
// output element, folding the broadcast inputs in input order on the device.
// Slots past the real input count repeat the first pointer; the kernel guards
// every fold step with the input count.
void LaunchFold(const std::vector<TensorView>& inputs, const TensorView& out, const std::string& context,
                const char* entry, const char* cache_key) {
    const int64_t out_count = ElementCount(out.dims, context);
    for (const TensorView& in : inputs) {
        if (in.data == nullptr) {
            throw InferenceException(context + ": null buffer");
        }
    }
    if (out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (out_count == 0) {
        return;
    }
    const StrideSet out_strides = Pack(Strides(out.dims), context);
    std::vector<std::vector<int64_t>> aligned;
    aligned.reserve(inputs.size());
    for (const TensorView& in : inputs) {
        aligned.push_back(AlignedStrides(in.dims, out.dims, context));
    }
    const StrideTable in_strides = PackTable(aligned, context);
    const cl_int rank = static_cast<cl_int>(out.dims.size());
    const cl_long count = out_count;
    const cl_int n = static_cast<cl_int>(inputs.size());
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);

    cl_kernel kernel = opencl_runtime::KernelFor(cache_key, kSource, entry);
    cl_mem mems[kMaxInputs];
    for (std::size_t k = 0; k < kMaxInputs; ++k) {
        const TensorView& slot = k < inputs.size() ? inputs[k] : inputs[0];
        mems[k] = reinterpret_cast<cl_mem>(const_cast<void*>(slot.data));
    }
    for (cl_uint k = 0; k < kMaxInputs; ++k) {
        CheckCl(clSetKernelArg(kernel, k, sizeof(cl_mem), &mems[k]), "clSetKernelArg", context);
    }
    CheckCl(clSetKernelArg(kernel, 8, sizeof(n), &n), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 9, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 10, sizeof(count), &count), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 11, sizeof(out_strides), &out_strides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 12, sizeof(in_strides), &in_strides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 13, sizeof(rank), &rank), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(out_count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

// Shared shape validation for the variadic rows: the NumPy broadcast of all
// inputs in order must equal the output shape.
void CheckVariadicShapes(const std::vector<TensorView>& inputs, const TensorView& out, const std::string& context) {
    std::vector<int64_t> acc = inputs[0].dims;
    (void)ElementCount(inputs[0].dims, context);
    for (std::size_t i = 1; i < inputs.size(); ++i) {
        (void)ElementCount(inputs[i].dims, context);
        acc = BroadcastDims(acc, inputs[i].dims, context);
    }
    if (acc != out.dims) {
        throw InferenceException(context + ": output shape does not match the broadcast of the inputs");
    }
}

} // namespace

void Clip(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Clip (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one to three inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != out.dtype) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (in.dtype == DataType::Float32) {
        RunClip<float>(inputs, outputs, context, -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::infinity(), DataType::Float32, "float32", "clip_f", "Clip/float");
    } else if (in.dtype == DataType::Int64) {
        RunClip<int64_t>(inputs, outputs, context, std::numeric_limits<int64_t>::min(),
                         std::numeric_limits<int64_t>::max(), DataType::Int64, "int64", "clip_i", "Clip/int64");
    } else {
        throw InferenceException(context + ": tensors must be float32");
    }
}

void Log(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Log (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    const int64_t count = ElementCount(out.dims, context);
    if (ElementCount(in.dims, context) != count) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (count == 0) {
        return;
    }
    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    const cl_long n = count;
    cl_kernel kernel = opencl_runtime::KernelFor("Log/float", kSource, "log_f");
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(n), &n), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

void Pow(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Pow (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two inputs and one output");
    }
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView& out = outputs[0];
    if (a.dtype != DataType::Float32 || b.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
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

    cl_kernel kernel = opencl_runtime::KernelFor("Pow/float", kSource, "pow_f");
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

void Max(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Max (node '" + node.name + "')";
    if (inputs.size() < 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected at least one input and one output");
    }
    if (inputs.size() > kMaxInputs) {
        throw InferenceException(context + ": input count exceeds the pilot device limit of 8");
    }
    const DataType dtype = inputs[0].dtype;
    if (dtype != DataType::Float32 && dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must share one float32 or int64 dtype");
    }
    for (const TensorView& in : inputs) {
        if (in.dtype != dtype) {
            throw InferenceException(context + ": tensors must share one float32 or int64 dtype");
        }
    }
    const TensorView& out = outputs[0];
    if (out.dtype != dtype) {
        throw InferenceException(context + ": tensors must share one float32 or int64 dtype");
    }
    CheckVariadicShapes(inputs, out, context);
    LaunchFold(inputs, out, context, dtype == DataType::Float32 ? "max_f" : "max_i",
               dtype == DataType::Float32 ? "Max/float" : "Max/int64");
}

void Min(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Min (node '" + node.name + "')";
    if (inputs.size() < 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected at least one input and one output");
    }
    if (inputs.size() > kMaxInputs) {
        throw InferenceException(context + ": input count exceeds the pilot device limit of 8");
    }
    const DataType dtype = inputs[0].dtype;
    if (dtype != DataType::Float32 && dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must share one float32 or int64 dtype");
    }
    for (const TensorView& in : inputs) {
        if (in.dtype != dtype) {
            throw InferenceException(context + ": tensors must share one float32 or int64 dtype");
        }
    }
    const TensorView& out = outputs[0];
    if (out.dtype != dtype) {
        throw InferenceException(context + ": tensors must share one float32 or int64 dtype");
    }
    CheckVariadicShapes(inputs, out, context);
    LaunchFold(inputs, out, context, dtype == DataType::Float32 ? "min_f" : "min_i",
               dtype == DataType::Float32 ? "Min/float" : "Min/int64");
}

void Or(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Or (node '" + node.name + "')";
    if (inputs.size() < 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected at least one input and one output");
    }
    if (inputs.size() > kMaxInputs) {
        throw InferenceException(context + ": input count exceeds the pilot device limit of 8");
    }
    for (const TensorView& in : inputs) {
        if (in.dtype != DataType::Bool) {
            throw InferenceException(context + ": inputs must be bool");
        }
    }
    const TensorView& out = outputs[0];
    if (out.dtype != DataType::Bool) {
        throw InferenceException(context + ": output must be bool");
    }
    CheckVariadicShapes(inputs, out, context);
    LaunchFold(inputs, out, context, "or_b", "Or/bool");
}

} // namespace opencl_kernels
} // namespace nert
