// Index-moving and integer rows on OpenCL: Split, GatherElements, TopK, Mod,
// and Tile.
//
// Mirrors the CPU oracles: Split copies one contiguous block per output row,
// GatherElements picks one element per index, TopK selects the K largest per
// row position in descending order with ties broken by the lower index, Mod
// applies the C++ truncating remainder under NumPy broadcast, and Tile reads
// each output coordinate modulo the input dim. Value-dependent checks (split
// sizes, gather indices, K, zero divisors, repeats) read the int64 inputs back
// to the host first so the thrown errors match the oracles.

#include "../axes_form.hpp"
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

StrideSet Pack(const std::vector<int64_t>& values, const std::string& context) {
    if (values.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the pilot limit of 16");
    }
    StrideSet packed{};
    for (std::size_t i = 0; i < values.size(); ++i) {
        packed.s[i] = values[i];
    }
    return packed;
}

// Reads an int64 tensor back to the host. The rows in this file validate
// values (sizes, indices, K, divisors, repeats) exactly like the CPU oracles,
// which need the numbers before any launch.
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

constexpr const char* kSplitSource = R"CLC(
__kernel void split_f32(__global const float* x, __global float* y, long count, long dim, long begin,
                        long block, long inner) {
    long l = get_global_id(0);
    if (l >= count) { return; }
    long o = l / block;
    long j = l - o * block;
    y[l] = x[(o * dim + begin) * inner + j];
}
__kernel void split_i64(__global const long* x, __global long* y, long count, long dim, long begin,
                        long block, long inner) {
    long l = get_global_id(0);
    if (l >= count) { return; }
    long o = l / block;
    long j = l - o * block;
    y[l] = x[(o * dim + begin) * inner + j];
}
)CLC";

constexpr const char* kGatherSource = R"CLC(
typedef struct { long s[16]; } StrideSet;
__kernel void gather_f32(__global const float* x, __global const long* idx, __global float* y, long count,
                         StrideSet idxStrides, StrideSet inStrides, int rank, int axis, long dim) {
    long i = get_global_id(0);
    if (i >= count) { return; }
    long remaining = i, offset = 0;
    for (int j = 0; j < rank; ++j) {
        long coord = remaining / idxStrides.s[j];
        remaining %= idxStrides.s[j];
        if (j != axis) { offset += coord * inStrides.s[j]; }
    }
    long k = idx[i];
    if (k < 0) { k += dim; }
    offset += k * inStrides.s[axis];
    y[i] = x[offset];
}
__kernel void gather_i64(__global const long* x, __global const long* idx, __global long* y, long count,
                         StrideSet idxStrides, StrideSet inStrides, int rank, int axis, long dim) {
    long i = get_global_id(0);
    if (i >= count) { return; }
    long remaining = i, offset = 0;
    for (int j = 0; j < rank; ++j) {
        long coord = remaining / idxStrides.s[j];
        remaining %= idxStrides.s[j];
        if (j != axis) { offset += coord * inStrides.s[j]; }
    }
    long k = idx[i];
    if (k < 0) { k += dim; }
    offset += k * inStrides.s[axis];
    y[i] = x[offset];
}
)CLC";

constexpr const char* kTopKSource = R"CLC(
__kernel void topk_f32(__global const float* x, __global float* values, __global long* indices, long count,
                       long dim, long k, long inner) {
    long i = get_global_id(0);
    if (i >= count) { return; }
    long o = i / inner;
    long n = i - o * inner;
    float prev_value = 0.0f;
    long prev_index = -1;
    for (long t = 0; t < k; ++t) {
        float best_value = 0.0f;
        long best_index = -1;
        int found = 0;
        for (long a = 0; a < dim; ++a) {
            const float v = x[(o * dim + a) * inner + n];
            const int eligible = (t == 0) || (v < prev_value) || (v == prev_value && a > prev_index);
            if (eligible != 0) {
                if (found == 0 || v > best_value || (v == best_value && a < best_index)) {
                    best_value = v;
                    best_index = a;
                    found = 1;
                }
            }
        }
        values[(o * k + t) * inner + n] = best_value;
        indices[(o * k + t) * inner + n] = best_index;
        prev_value = best_value;
        prev_index = best_index;
    }
}
)CLC";

constexpr const char* kModSource = R"CLC(
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
__kernel void mod_i64(__global const long* a, __global const long* b, __global long* y, long count,
                      StrideSet outStrides, StrideSet aStrides, StrideSet bStrides, int rank) {
    DECODE_OFFSETS(c)
    y[i] = a[ao] % b[bo];
}
)CLC";

constexpr const char* kTileSource = R"CLC(
typedef struct { long s[16]; } StrideSet;
__kernel void tile_f32(__global const float* x, __global float* y, long count, StrideSet outStrides,
                       StrideSet inStrides, StrideSet inDims, int rank) {
    long i = get_global_id(0);
    if (i >= count) { return; }
    long remaining = i, offset = 0;
    for (int j = 0; j < rank; ++j) {
        long coord = remaining / outStrides.s[j];
        remaining %= outStrides.s[j];
        offset += (coord % inDims.s[j]) * inStrides.s[j];
    }
    y[i] = x[offset];
}
__kernel void tile_i64(__global const long* x, __global long* y, long count, StrideSet outStrides,
                       StrideSet inStrides, StrideSet inDims, int rank) {
    long i = get_global_id(0);
    if (i >= count) { return; }
    long remaining = i, offset = 0;
    for (int j = 0; j < rank; ++j) {
        long coord = remaining / outStrides.s[j];
        remaining %= outStrides.s[j];
        offset += (coord % inDims.s[j]) * inStrides.s[j];
    }
    y[i] = x[offset];
}
__kernel void tile_b(__global const uchar* x, __global uchar* y, long count, StrideSet outStrides,
                     StrideSet inStrides, StrideSet inDims, int rank) {
    long i = get_global_id(0);
    if (i >= count) { return; }
    long remaining = i, offset = 0;
    for (int j = 0; j < rank; ++j) {
        long coord = remaining / outStrides.s[j];
        remaining %= outStrides.s[j];
        offset += (coord % inDims.s[j]) * inStrides.s[j];
    }
    y[i] = x[offset];
}
)CLC";

} // namespace

void Split(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Split (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 2 || outputs.empty()) {
        throw InferenceException(context + ": expected one or two inputs and at least one output");
    }
    const TensorView& in = inputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be float32 or int64");
    }
    for (const TensorView& view : outputs) {
        if (view.dtype != in.dtype) {
            throw InferenceException(context + ": output dtype must match the input dtype");
        }
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (rank == 0) {
        throw InferenceException(context + ": Split axis is out of range");
    }
    int64_t axis = IntAttribute(node, "axis", 0, context);
    if (axis < -rank || axis >= rank) {
        throw InferenceException(context + ": Split axis is out of range");
    }
    if (axis < 0) {
        axis += rank;
    }
    const int64_t dim = in.dims[static_cast<std::size_t>(axis)];

    std::vector<int64_t> parts;
    bool parts_supplied = false;
    const SplitForm split_form = ResolveSplitForm(node);
    if (split_form.source == AxesSource::Attribute) {
        if (split_form.split_attribute != nullptr) {
            const std::vector<int64_t>* values = NodeAttributeIntList(split_form.split_attribute);
            if (values == nullptr) {
                throw InferenceException(context + ": attribute 'split' must be an integer list");
            }
            parts = *values;
            parts_supplied = true;
        }
    } else if (inputs.size() == 2 && inputs[1].dtype != DataType::Unknown) {
        const TensorView& sizes = inputs[1];
        if (sizes.dtype != DataType::Int64) {
            throw InferenceException(context + ": split sizes must be int64");
        }
        const int64_t count = ElementCount(sizes.dims, context);
        if (count > 0 && sizes.data == nullptr) {
            throw InferenceException(context + ": null split-sizes buffer");
        }
        parts = ReadInt64(sizes, count, context);
        parts_supplied = true;
    }
    const auto default_parts = DefaultSplitParts(node, dim, outputs.size(), parts_supplied, context);
    if (parts_supplied) {
        if (parts.size() != outputs.size()) {
            throw InferenceException(context + ": Split split sizes must match the output count");
        }
        int64_t sum = 0;
        for (const int64_t v : parts) {
            if (v < 0) {
                throw InferenceException(context + ": Split split sizes must be non-negative");
            }
            sum += v;
        }
        if (sum != dim) {
            throw InferenceException(context + ": Split split sizes must sum to the axis dim");
        }
    } else {
        parts = default_parts;
    }

    for (std::size_t i = 0; i < outputs.size(); ++i) {
        std::vector<int64_t> want = in.dims;
        want[static_cast<std::size_t>(axis)] = parts[i];
        if (outputs[i].dims != want) {
            throw InferenceException(context + ": output shape does not match the split sizes");
        }
        if (ElementCount(outputs[i].dims, context) > 0 && outputs[i].data == nullptr) {
            throw InferenceException(context + ": null output buffer");
        }
    }
    (void)ElementCount(in.dims, context);
    if (in.data == nullptr) {
        throw InferenceException(context + ": null input buffer");
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= in.dims[static_cast<std::size_t>(i)];
    }
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= in.dims[static_cast<std::size_t>(i)];
    }

    const bool is_float = in.dtype == DataType::Float32;
    cl_kernel kernel = is_float ? opencl_runtime::KernelFor("Split/float", kSplitSource, "split_f32")
                                : opencl_runtime::KernelFor("Split/int64", kSplitSource, "split_i64");
    cl_mem xm = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    int64_t begin = 0;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const int64_t extent = parts[static_cast<std::size_t>(i)];
        const int64_t count = outer * extent * inner;
        if (count > 0 && outputs[i].data != nullptr) {
            cl_mem ym = reinterpret_cast<cl_mem>(outputs[i].data);
            const cl_long count_arg = count;
            const cl_long dim_arg = dim;
            const cl_long begin_arg = begin;
            const cl_long block_arg = extent * inner;
            const cl_long inner_arg = inner;
            CheckCl(clSetKernelArg(kernel, 0, sizeof(xm), &xm), "clSetKernelArg", context);
            CheckCl(clSetKernelArg(kernel, 1, sizeof(ym), &ym), "clSetKernelArg", context);
            CheckCl(clSetKernelArg(kernel, 2, sizeof(count_arg), &count_arg), "clSetKernelArg", context);
            CheckCl(clSetKernelArg(kernel, 3, sizeof(dim_arg), &dim_arg), "clSetKernelArg", context);
            CheckCl(clSetKernelArg(kernel, 4, sizeof(begin_arg), &begin_arg), "clSetKernelArg", context);
            CheckCl(clSetKernelArg(kernel, 5, sizeof(block_arg), &block_arg), "clSetKernelArg", context);
            CheckCl(clSetKernelArg(kernel, 6, sizeof(inner_arg), &inner_arg), "clSetKernelArg", context);
            const std::size_t global = static_cast<std::size_t>(count);
            CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr,
                                           nullptr),
                    "clEnqueueNDRangeKernel", context);
        }
        begin += extent;
    }
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

void GatherElements(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "GatherElements (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected data and indices inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& indices = inputs[1];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be float32 or int64");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    if (indices.dtype != DataType::Int64) {
        throw InferenceException(context + ": GatherElements indices must be int64");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (static_cast<int64_t>(indices.dims.size()) != rank) {
        throw InferenceException(context + ": GatherElements data and indices must share rank");
    }
    const int64_t axis = IntAttribute(node, "axis", 1, context);
    if (rank == 0 || axis < -rank || axis >= rank) {
        throw InferenceException(context + ": GatherElements axis is out of range");
    }
    const int64_t norm_axis = axis < 0 ? axis + rank : axis;
    if (out.dims != indices.dims) {
        throw InferenceException(context + ": output shape must match the indices shape");
    }

    const int64_t count = ElementCount(indices.dims, context);
    (void)ElementCount(in.dims, context);
    (void)ElementCount(out.dims, context);
    if (indices.data == nullptr) {
        throw InferenceException(context + ": null indices buffer");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const int64_t dim = in.dims[static_cast<std::size_t>(norm_axis)];
    // The indices live on the device; read them back so the wrap and range
    // checks match the CPU oracle before the gather launches.
    const std::vector<int64_t> idx = ReadInt64(indices, count, context);
    const std::vector<int64_t> idx_strides = Strides(indices.dims);
    for (int64_t i = 0; i < count; ++i) {
        int64_t remaining = i;
        for (int64_t j = 0; j < rank; ++j) {
            const std::size_t uj = static_cast<std::size_t>(j);
            const int64_t coord = remaining / idx_strides[uj];
            remaining %= idx_strides[uj];
            if (j != norm_axis && coord >= in.dims[uj]) {
                throw InferenceException(context + ": GatherElements index is out of range");
            }
        }
        int64_t k = idx[static_cast<std::size_t>(i)];
        if (k < 0) {
            k += dim;
        }
        if (k < 0 || k >= dim) {
            throw InferenceException(context + ": GatherElements index is out of range");
        }
    }

    const bool is_float = in.dtype == DataType::Float32;
    cl_kernel kernel = is_float ? opencl_runtime::KernelFor("GatherElements/float", kGatherSource, "gather_f32")
                                : opencl_runtime::KernelFor("GatherElements/int64", kGatherSource, "gather_i64");
    cl_mem xm = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem idxm = reinterpret_cast<cl_mem>(const_cast<void*>(indices.data));
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    const StrideSet idx_strides_arg = Pack(idx_strides, context);
    const StrideSet in_strides_arg = Pack(Strides(in.dims), context);
    const cl_int rank_arg = static_cast<cl_int>(rank);
    const cl_int axis_arg = static_cast<cl_int>(norm_axis);
    const cl_long count_arg = count;
    const cl_long dim_arg = dim;
    CheckCl(clSetKernelArg(kernel, 0, sizeof(xm), &xm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(idxm), &idxm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(count_arg), &count_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(idx_strides_arg), &idx_strides_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(in_strides_arg), &in_strides_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(rank_arg), &rank_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 7, sizeof(axis_arg), &axis_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 8, sizeof(dim_arg), &dim_arg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

void TopK(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "TopK (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 2) {
        throw InferenceException(context + ": expected an input, a K input, and values/indices outputs");
    }
    const TensorView& in = inputs[0];
    const TensorView& k_view = inputs[1];
    const TensorView& values = outputs[0];
    const TensorView& indices = outputs[1];
    if (in.dtype != DataType::Float32) {
        throw InferenceException(context + ": TopK input must be float32");
    }
    if (values.dtype != DataType::Float32) {
        throw InferenceException(context + ": TopK values must be float32");
    }
    if (indices.dtype != DataType::Int64) {
        throw InferenceException(context + ": TopK indices must be int64");
    }
    if (k_view.dtype != DataType::Int64) {
        throw InferenceException(context + ": TopK requires a K input");
    }
    if (IntAttribute(node, "largest", 1, context) != 1) {
        throw InferenceException(context + ": TopK 'largest' must be 1 here");
    }
    if (IntAttribute(node, "sorted", 1, context) != 1) {
        throw InferenceException(context + ": TopK 'sorted' must be 1 here");
    }

    if (ElementCount(k_view.dims, context) != 1) {
        throw InferenceException(context + ": TopK K input must be a scalar Int64 constant");
    }
    if (k_view.data == nullptr) {
        throw InferenceException(context + ": null K buffer");
    }
    // K is a scalar on the device; the host needs it to shape and bound the
    // selection exactly like the CPU oracle.
    const int64_t k = ReadInt64(k_view, 1, context)[0];
    if (k <= 0) {
        throw InferenceException(context + ": TopK K must be positive");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    const int64_t axis = IntAttribute(node, "axis", -1, context);
    if (rank == 0 || axis < -rank || axis >= rank) {
        throw InferenceException(context + ": TopK axis is out of range");
    }
    const int64_t norm_axis = axis < 0 ? axis + rank : axis;
    const int64_t dim = in.dims[static_cast<std::size_t>(norm_axis)];
    if (k > dim) {
        throw InferenceException(context + ": TopK K exceeds the axis dim");
    }
    std::vector<int64_t> expected = in.dims;
    expected[static_cast<std::size_t>(norm_axis)] = k;
    if (values.dims != expected || indices.dims != expected) {
        throw InferenceException(context + ": output shapes do not match the selected K");
    }

    const int64_t values_count = ElementCount(values.dims, context);
    (void)ElementCount(in.dims, context);
    if (values_count > 0 && (in.data == nullptr || values.data == nullptr || indices.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < norm_axis; ++i) {
        outer *= in.dims[static_cast<std::size_t>(i)];
    }
    int64_t inner = 1;
    for (int64_t i = norm_axis + 1; i < rank; ++i) {
        inner *= in.dims[static_cast<std::size_t>(i)];
    }

    cl_kernel kernel = opencl_runtime::KernelFor("TopK", kTopKSource, "topk_f32");
    cl_mem xm = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem ym = reinterpret_cast<cl_mem>(values.data);
    cl_mem iym = reinterpret_cast<cl_mem>(indices.data);
    const cl_long count_arg = outer * inner;
    const cl_long dim_arg = dim;
    const cl_long k_arg = k;
    const cl_long inner_arg = inner;
    CheckCl(clSetKernelArg(kernel, 0, sizeof(xm), &xm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(iym), &iym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(count_arg), &count_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(dim_arg), &dim_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(k_arg), &k_arg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(inner_arg), &inner_arg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(count_arg);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

void Mod(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Mod (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two inputs and one output");
    }
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "fmod") {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'fmod' must be an integer");
            }
            if (*value != 0) {
                throw InferenceException(context + ": only fmod=0 (integer remainder) is supported");
            }
        }
    }
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView& out = outputs[0];
    if (a.dtype != DataType::Int64 || b.dtype != DataType::Int64 || out.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be int64");
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

    // Every divisor element is read at least once under the broadcast, so a
    // host-side scan rejects zeros exactly like the oracle's per-element check.
    const std::vector<int64_t> divisors = ReadInt64(b, ElementCount(b.dims, context), context);
    for (const int64_t v : divisors) {
        if (v == 0) {
            throw InferenceException(context + ": division by zero");
        }
    }

    const StrideSet out_strides = Pack(Strides(out.dims), context);
    const StrideSet a_strides = Pack(AlignedStrides(a.dims, out.dims, context), context);
    const StrideSet b_strides = Pack(AlignedStrides(b.dims, out.dims, context), context);
    const cl_int rank = static_cast<cl_int>(out.dims.size());
    const cl_long count = out_count;
    cl_kernel kernel = opencl_runtime::KernelFor("Mod", kModSource, "mod_i64");
    cl_mem am = reinterpret_cast<cl_mem>(const_cast<void*>(a.data));
    cl_mem bm = reinterpret_cast<cl_mem>(const_cast<void*>(b.data));
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
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

void Tile(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Tile (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected a data input, a repeats input, and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& repeats = inputs[1];
    const TensorView& out = outputs[0];
    if (in.dtype != out.dtype ||
        (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool)) {
        throw InferenceException(context + ": data must share one float32, int64, or bool dtype");
    }
    if (repeats.dtype != DataType::Int64 || repeats.dims.size() != 1 ||
        repeats.dims[0] != static_cast<int64_t>(in.dims.size())) {
        throw InferenceException(context + ": repeats must be a rank-1 int64 tensor with one entry per input axis");
    }
    (void)ElementCount(in.dims, context);
    const int64_t out_count = ElementCount(out.dims, context);
    if (in.data == nullptr || repeats.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    // The repeats live on the device; read them back so the shape rule matches
    // the CPU oracle before the launch.
    const std::vector<int64_t> reps = ReadInt64(repeats, static_cast<int64_t>(in.dims.size()), context);
    if (out.dims.size() != in.dims.size()) {
        throw InferenceException(context + ": output rank does not match the input rank");
    }
    for (std::size_t j = 0; j < in.dims.size(); ++j) {
        if (reps[j] <= 0 || out.dims[j] != in.dims[j] * reps[j]) {
            throw InferenceException(context + ": output shape does not match input dims times repeats");
        }
    }

    const StrideSet out_strides = Pack(Strides(out.dims), context);
    const StrideSet in_strides = Pack(Strides(in.dims), context);
    const StrideSet in_dims = Pack(in.dims, context);
    const cl_int rank = static_cast<cl_int>(in.dims.size());
    const cl_long count = out_count;
    const char* entry = nullptr;
    const char* cache_key = nullptr;
    if (in.dtype == DataType::Float32) {
        entry = "tile_f32";
        cache_key = "Tile/float";
    } else if (in.dtype == DataType::Int64) {
        entry = "tile_i64";
        cache_key = "Tile/int64";
    } else {
        entry = "tile_b";
        cache_key = "Tile/bool";
    }
    cl_kernel kernel = opencl_runtime::KernelFor(cache_key, kTileSource, entry);
    cl_mem xm = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(xm), &xm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(count), &count), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(out_strides), &out_strides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(in_strides), &in_strides), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(in_dims), &in_dims), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(rank), &rank), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(out_count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
