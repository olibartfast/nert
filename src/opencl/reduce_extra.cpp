// ReduceSum and ReduceMax on OpenCL: reduction over the requested axes.
//
// Mirrors the CPU oracles: ReduceSum takes float32 or int64 data (the
// output dtype must match the input), and ReduceMax is float32 only;
// the axes arrive as an optional int64 input (absent or Unknown means
// every axis, subject to `noop_with_empty_axes` for the sum's identity
// copy); `keepdims` defaults to 1; negative axes are normalized and
// out-of-range or duplicate axes are rejected. The float32 sum
// accumulates in double and rounds once to float32 (the device must
// report double-precision support, cl_khr_fp64); the int64 sum
// accumulates in long and is exact, with its own kernel and cache key;
// ReduceMax compares floats directly. One work-item per output element
// gathers over the reduced coordinates in row-major order; index math
// is capped at rank 16 like the neighbour rows. The OpenCL C sources
// compile once per process via the shared runtime.

#include "../reduce_axes.hpp"
#include "nert/device.hpp"
#include "runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

namespace nert {
namespace opencl_kernels {
namespace {

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

// Reads an int64 attribute, or returns the fallback when the attribute is absent.
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

constexpr int kMaxRank = 16;

// Everything the kernels need to map one output element onto its reduced
// input span. Passed by value as one kernel argument; every member is an
// 8-byte integer, so the host and device layouts agree.
struct ReduceParams {
    cl_long rank;
    cl_long numReduced;
    cl_long reducedCount;
    cl_long outCount;
    cl_long outStride[kMaxRank];
    cl_long inStride[kMaxRank];
    cl_long outAxis[kMaxRank]; // per input axis: output axis, or -1 when reduced
    cl_long redDims[kMaxRank];
    cl_long redStride[kMaxRank];
    cl_long redInStride[kMaxRank];
};
static_assert(sizeof(ReduceParams) == 4 * sizeof(cl_long) + 6 * kMaxRank * sizeof(cl_long),
              "device struct layout must match");

constexpr const char* kSumSource = R"CLC(
typedef struct {
    long rank;
    long numReduced;
    long reducedCount;
    long outCount;
    long outStride[16];
    long inStride[16];
    long outAxis[16];
    long redDims[16];
    long redStride[16];
    long redInStride[16];
} ReduceParams;

// One work-item per output element: walk the surviving coordinates for
// the base input offset, then gather the reduced span in row-major
// order, accumulating in double and rounding once to float32.
__kernel void reduce_sum(__global const float* x, __global float* y, ReduceParams p) {
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
        y[o] = (float)acc;
    }
}
)CLC";

// The int64 variant of the sum kernel: same index math, but the
// accumulator is a long, so integer sums are exact and no fp64 support
// is required. Compiled under its own cache key.
constexpr const char* kSumInt64Source = R"CLC(
typedef struct {
    long rank;
    long numReduced;
    long reducedCount;
    long outCount;
    long outStride[16];
    long inStride[16];
    long outAxis[16];
    long redDims[16];
    long redStride[16];
    long redInStride[16];
} ReduceParams;

// One work-item per output element: walk the surviving coordinates for
// the base input offset, then gather the reduced span in row-major
// order, summing the int64 values exactly.
__kernel void reduce_sum_i64(__global const long* x, __global long* y, ReduceParams p) {
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
        long acc = 0;
        for (long r = 0; r < p.reducedCount; ++r) {
            long off = base;
            long t = r;
            for (long j = 0; j < p.numReduced; ++j) {
                long coord = t / p.redStride[j];
                t -= coord * p.redStride[j];
                off += coord * p.redInStride[j];
            }
            acc += x[off];
        }
        y[o] = acc;
    }
}
)CLC";

constexpr const char* kMaxSource = R"CLC(
typedef struct {
    long rank;
    long numReduced;
    long reducedCount;
    long outCount;
    long outStride[16];
    long inStride[16];
    long outAxis[16];
    long redDims[16];
    long redStride[16];
    long redInStride[16];
} ReduceParams;

// One work-item per output element: gather the reduced span and keep the
// largest float, starting from float's lowest value like the CPU oracle.
__kernel void reduce_max(__global const float* x, __global float* y, ReduceParams p) {
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
        float best = -FLT_MAX;
        for (long r = 0; r < p.reducedCount; ++r) {
            long off = base;
            long t = r;
            for (long j = 0; j < p.numReduced; ++j) {
                long coord = t / p.redStride[j];
                t -= coord * p.redStride[j];
                off += coord * p.redInStride[j];
            }
            if (x[off] > best) {
                best = x[off];
            }
        }
        y[o] = best;
    }
}
)CLC";

// Shared validation and index planning for both reductions. `allowNoop`
// selects the ReduceSum oracle's noop_with_empty_axes identity (the
// ReduceMax oracle has no such attribute handling).
struct ReducePlan {
    bool identity = false;
    bool int64Row = false;
    std::vector<int64_t> inDims;
    std::vector<int64_t> outDims;
    std::vector<int64_t> inStride;
    std::vector<int64_t> outStride;
    std::vector<int64_t> outAxis;
    std::vector<int64_t> redDims;
    std::vector<int64_t> redStride;
    std::vector<int64_t> redInStride;
    int64_t inCount = 0;
    int64_t outCount = 0;
    int64_t reducedCount = 0;
};

ReducePlan PlanReduction(const Node& node, const std::vector<TensorView>& inputs,
                         const std::vector<TensorView>& outputs, bool allowNoop, bool allowInt64,
                         bool version_aware_axes, const std::string& context) {
    if (inputs.empty() || inputs.size() > 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one or two inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    // The sum row dispatches on the data dtype: float32 or int64, with
    // the output dtype matching the input. The max row is float32 only,
    // like its oracle.
    const bool int64Row = allowInt64 && in.dtype == DataType::Int64;
    if (in.dtype != out.dtype || (!int64Row && in.dtype != DataType::Float32)) {
        throw InferenceException(
            context + (allowInt64 ? ": data tensors must be float32 or int64" : ": data tensors must be float32"));
    }
    const std::vector<int64_t>& inDims = in.dims;
    const int64_t rank = static_cast<int64_t>(inDims.size());
    if (rank > kMaxRank) {
        throw InferenceException(context + ": input rank must not exceed 16");
    }
    ReducePlan plan;
    plan.int64Row = int64Row;
    plan.inDims = inDims;
    plan.inCount = ElementCount(inDims, context);
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const bool keepdims = IntAttribute(node, "keepdims", 1, context) != 0;
    const bool noopEmpty = allowNoop && IntAttribute(node, "noop_with_empty_axes", 0, context) != 0;
    const ReduceAxes resolved_axes = ResolveReduceAxes(node);

    // An absent, Unknown, or empty axes input means every axis unless
    // noop_with_empty_axes asks for identity (ReduceSum; ReduceMax's noop is
    // rejected at load as a tracked gap). ReduceMax additionally accepts the
    // pre-18 attribute form in place.
    std::vector<int64_t> axes;
    if (version_aware_axes && resolved_axes.form == ReduceAxesForm::Attribute) {
        if (resolved_axes.axes_attribute != nullptr) {
            const std::vector<int64_t>* values = ReduceAttributeIntList(resolved_axes.axes_attribute);
            if (values == nullptr) {
                throw InferenceException(context + ": attribute 'axes' must be an integer list");
            }
            axes = *values;
        }
    } else if (inputs.size() == 2 && inputs[1].dtype != DataType::Unknown) {
        const TensorView& axesView = inputs[1];
        if (axesView.dtype != DataType::Int64) {
            throw InferenceException(context + ": axes tensor must be int64");
        }
        const int64_t axesCount = ReduceAxesElementCount(axesView.dims, context);
        if (axesCount > 0) {
            if (axesView.data == nullptr) {
                throw InferenceException(context + ": null axes buffer");
            }
            // The axes tensor lives in device memory (a cl_mem handle);
            // read the small values back so the host can validate them.
            std::vector<int64_t> values(static_cast<std::size_t>(axesCount));
            cl_mem am = reinterpret_cast<cl_mem>(const_cast<void*>(axesView.data));
            CheckCl(clEnqueueReadBuffer(opencl_runtime::Queue(), am, CL_TRUE, 0,
                                        static_cast<std::size_t>(axesCount) * sizeof(int64_t), values.data(), 0,
                                        nullptr, nullptr),
                    "clEnqueueReadBuffer", context);
            axes = std::move(values);
        }
    }
    if (axes.empty()) {
        if (noopEmpty) {
            plan.identity = true;
        } else {
            for (int64_t i = 0; i < rank; ++i) {
                axes.push_back(i);
            }
        }
    }
    if (plan.identity) {
        if (out.dims != inDims) {
            throw InferenceException(context + ": output shape does not match the input");
        }
        plan.outDims = inDims;
        plan.outCount = plan.inCount;
        return plan;
    }

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
    for (int64_t i = 0; i < rank; ++i) {
        if (reduced[static_cast<std::size_t>(i)]) {
            if (keepdims) {
                plan.outDims.push_back(1);
            }
        } else {
            plan.outDims.push_back(inDims[static_cast<std::size_t>(i)]);
        }
    }
    if (out.dims != plan.outDims) {
        throw InferenceException(context + ": output shape does not match the reduced axes");
    }
    plan.outCount = ElementCount(plan.outDims, context);
    if (plan.outCount == 0) {
        return plan;
    }

    plan.inStride.assign(inDims.size(), 1);
    for (std::size_t i = inDims.size(); i-- > 1;) {
        plan.inStride[i - 1] = plan.inStride[i] * inDims[i];
    }
    plan.outStride.assign(plan.outDims.size(), 1);
    for (std::size_t i = plan.outDims.size(); i-- > 1;) {
        plan.outStride[i - 1] = plan.outStride[i] * plan.outDims[i];
    }
    plan.outAxis.assign(static_cast<std::size_t>(rank), -1);
    int64_t nextOutAxis = 0;
    for (int64_t i = 0; i < rank; ++i) {
        if (!reduced[static_cast<std::size_t>(i)]) {
            plan.outAxis[static_cast<std::size_t>(i)] = keepdims ? i : nextOutAxis;
            ++nextOutAxis;
        }
    }
    std::vector<int64_t> redAxes;
    for (int64_t i = 0; i < rank; ++i) {
        if (reduced[static_cast<std::size_t>(i)]) {
            redAxes.push_back(i);
        }
    }
    plan.redDims.assign(redAxes.size(), 0);
    plan.redStride.assign(redAxes.size(), 1);
    plan.redInStride.assign(redAxes.size(), 1);
    int64_t reducedCount = 1;
    for (std::size_t j = redAxes.size(); j-- > 0;) {
        plan.redStride[j] = reducedCount;
        plan.redDims[j] = inDims[static_cast<std::size_t>(redAxes[j])];
        plan.redInStride[j] = plan.inStride[static_cast<std::size_t>(redAxes[j])];
        reducedCount *= plan.redDims[j];
    }
    plan.reducedCount = reducedCount;
    return plan;
}

ReduceParams MakeParams(const ReducePlan& plan) {
    ReduceParams params{};
    params.rank = static_cast<cl_long>(plan.inDims.size());
    params.numReduced = static_cast<cl_long>(plan.redDims.size());
    params.reducedCount = static_cast<cl_long>(plan.reducedCount);
    params.outCount = static_cast<cl_long>(plan.outCount);
    for (std::size_t i = 0; i < plan.outStride.size() && i < kMaxRank; ++i) {
        params.outStride[i] = plan.outStride[i];
    }
    for (std::size_t i = 0; i < plan.inStride.size() && i < kMaxRank; ++i) {
        params.inStride[i] = plan.inStride[i];
    }
    for (std::size_t i = 0; i < plan.outAxis.size() && i < kMaxRank; ++i) {
        params.outAxis[i] = plan.outAxis[i];
    }
    for (std::size_t i = 0; i < plan.redDims.size() && i < kMaxRank; ++i) {
        params.redDims[i] = plan.redDims[i];
        params.redStride[i] = plan.redStride[i];
        params.redInStride[i] = plan.redInStride[i];
    }
    return params;
}

void LaunchReduction(const std::string& op, const char* source, const char* entry, const ReduceParams& params,
                     const TensorView& in, const TensorView& out, const std::string& context) {
    cl_mem xm = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    cl_kernel kernel = opencl_runtime::KernelFor(op, source, entry);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(xm), &xm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(params), &params), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(params.outCount);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace

void ReduceSum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ReduceSum (node '" + node.name + "')";
    ReducePlan plan = PlanReduction(node, inputs, outputs, /*allowNoop=*/true, /*allowInt64=*/true,
                                    /*version_aware_axes=*/false, context);
    if (plan.outCount == 0) {
        return;
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (plan.identity) {
        // Identity: copy the device buffer (no accumulation, no fp64 need).
        cl_mem xm = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
        cl_mem ym = reinterpret_cast<cl_mem>(out.data);
        const std::size_t bytes =
            static_cast<std::size_t>(plan.inCount) * (plan.int64Row ? sizeof(int64_t) : sizeof(float));
        CheckCl(clEnqueueCopyBuffer(opencl_runtime::Queue(), xm, ym, 0, 0, bytes, 0, nullptr, nullptr),
                "clEnqueueCopyBuffer", context);
        CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
        return;
    }
    if (!plan.int64Row) {
        // The float32 kernel accumulates in double; OpenCL fp64 is
        // optional, so fail with the device capability named instead of a
        // build error. The int64 kernel needs no double support.
        cl_device_id device = nullptr;
        CheckCl(clGetContextInfo(opencl_runtime::Context(), CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr),
                "clGetContextInfo", context);
        cl_device_fp_config fp64 = 0;
        CheckCl(clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp64), &fp64, nullptr), "clGetDeviceInfo",
                context);
        if (fp64 == 0) {
            throw InferenceException(context + ": device has no double-precision support (cl_khr_fp64 is required)");
        }
    }
    const ReduceParams params = MakeParams(plan);
    if (plan.int64Row) {
        LaunchReduction("ReduceSum/int64", kSumInt64Source, "reduce_sum_i64", params, in, out, context);
    } else {
        LaunchReduction("ReduceSum", kSumSource, "reduce_sum", params, in, out, context);
    }
}

void ReduceMax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ReduceMax (node '" + node.name + "')";
    ReducePlan plan = PlanReduction(node, inputs, outputs, /*allowNoop=*/false, /*allowInt64=*/false,
                                    /*version_aware_axes=*/true, context);
    if (plan.outCount == 0) {
        return;
    }
    const ReduceParams params = MakeParams(plan);
    LaunchReduction("ReduceMax", kMaxSource, "reduce_max", params, inputs[0], outputs[0], context);
}

} // namespace opencl_kernels
} // namespace nert
