// InstanceNormalization, Range, and Pad on OpenCL.
//
// Mirrors the CPU oracles (src/kernels): InstanceNormalization normalizes
// each (n, c) row of a float32 [N, C, ...] input with statistics accumulated
// in double, so the device must report double-precision support
// (cl_khr_fp64); Range reads its start/limit/delta scalars back from device
// memory, derives the output length on the host like the oracle, and fills
// the 1-D output one element per work-item; Pad reads its int64
// paddings/axes and scalar fill value back from device memory and maps
// reflect/edge positions like the oracle's MapPosition. Kernel sources
// compile once per process via the shared runtime, one cache key per op
// and dtype.

#include "nert/device.hpp"
#include "runtime.hpp"

#include <cmath>
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

void CheckCl(cl_int status, const std::string& what, const std::string& context) {
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

float FloatAttribute(const Node& node, const std::string& name, float fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const float* value = std::get_if<float>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("InstanceNormalization: attribute '" + name + "' must be a float (node '" +
                                         node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

// A per-channel parameter (scale or B): rank-1 float32 of length `channels`,
// living in device memory.
cl_mem ChannelParam(const TensorView& view, int64_t channels, const std::string& context, const char* what) {
    if (view.dtype != DataType::Float32 || view.data == nullptr) {
        throw InferenceException(context + ": " + what + " must be a float32 tensor");
    }
    if (view.dims.size() != 1 || view.dims[0] != channels) {
        throw InferenceException(context + ": " + what + " must be rank-1 with one value per channel");
    }
    return reinterpret_cast<cl_mem>(view.data);
}

// Validates a scalar operand like the oracle's ReadScalar and reads its
// single element back from device memory.
template <typename T>
T ReadDeviceScalar(const TensorView& view, DataType dtype, const char* what, const std::string& context) {
    if (view.dtype != dtype) {
        throw InferenceException(context + ": " + what + " dtype must match the other operands");
    }
    int64_t count = 1;
    for (int64_t dim : view.dims) {
        count *= dim;
    }
    if (count != 1) {
        throw InferenceException(context + ": " + what + " must be a scalar");
    }
    if (view.data == nullptr) {
        throw InferenceException(context + ": null " + what + " buffer");
    }
    T value{};
    cl_mem mem = reinterpret_cast<cl_mem>(view.data);
    CheckCl(clEnqueueReadBuffer(opencl_runtime::Queue(), mem, CL_TRUE, 0, sizeof(T), &value, 0, nullptr, nullptr),
            std::string("clEnqueueReadBuffer (") + what + ")", context);
    return value;
}

// Validates an int64 operand like the oracle's ReadInt64s and reads its
// elements back from device memory.
std::vector<int64_t> ReadDeviceInt64s(const TensorView& view, const char* what, const std::string& context) {
    if (view.dtype != DataType::Int64) {
        throw InferenceException(context + ": " + what + " must be int64");
    }
    int64_t count = 1;
    for (int64_t dim : view.dims) {
        count *= dim;
    }
    if (count > 0 && view.data == nullptr) {
        throw InferenceException(context + ": null " + what + " buffer");
    }
    std::vector<int64_t> values(static_cast<std::size_t>(count), 0);
    if (count > 0) {
        cl_mem mem = reinterpret_cast<cl_mem>(view.data);
        CheckCl(clEnqueueReadBuffer(opencl_runtime::Queue(), mem, CL_TRUE, 0,
                                    static_cast<std::size_t>(count) * sizeof(int64_t), values.data(), 0, nullptr,
                                    nullptr),
                std::string("clEnqueueReadBuffer (") + what + ")", context);
    }
    return values;
}

constexpr const char* kInstnormSource = R"CLC(
__kernel void instnorm_f(__global const float* x, __global const float* scale, __global const float* bias,
                         __global float* y, long rows, long channels, long spatial, double epsilon) {
    long r = get_global_id(0);
    if (r < rows) {
        long c = r % channels;
        __global const float* in = x + r * spatial;
        double sum = 0.0;
        for (long k = 0; k < spatial; ++k) {
            sum += (double)in[k];
        }
        double mean = sum / (double)spatial;
        double squares = 0.0;
        for (long k = 0; k < spatial; ++k) {
            double centered = (double)in[k] - mean;
            squares += centered * centered;
        }
        double invStd = 1.0 / sqrt(squares / (double)spatial + epsilon);
        double s = (double)scale[c];
        double b = (double)bias[c];
        for (long k = 0; k < spatial; ++k) {
            y[r * spatial + k] = (float)(((double)in[k] - mean) * invStd * s + b);
        }
    }
}
)CLC";

constexpr const char* kRangeInt64Source = R"CLC(
__kernel void range_i64(__global long* y, long start, long delta, long count) {
    long i = get_global_id(0);
    if (i < count) {
        y[i] = start + i * delta;
    }
}
)CLC";

constexpr const char* kRangeFloatSource = R"CLC(
#pragma OPENCL FP_CONTRACT OFF
__kernel void range_f32(__global float* y, float start, float delta, long count) {
    long i = get_global_id(0);
    if (i < count) {
        float scaled = (float)i * delta;
        y[i] = start + scaled;
    }
}
)CLC";

// Pad kernel metadata: kMaxRank (16) slots each, indexed by axis, for the
// begin pads [0..15], input dims [16..31], input strides [32..47], and
// output strides [48..63]. The mode encodes 0 constant, 1 reflect, 2 edge;
// MapPadPosition mirrors the oracle's MapPosition exactly.
constexpr const char* kPadFloatSource = R"CLC(
long MapPadPosition(long pos, long dim, int mode) {
    if (pos >= 0 && pos < dim) {
        return pos;
    }
    if (mode == 0) {
        return -1;
    }
    if (mode == 2) {
        return pos < 0 ? 0 : dim - 1;
    }
    if (dim == 1) {
        return 0;
    }
    long period = 2 * (dim - 1);
    long m = pos % period;
    if (m < 0) {
        m += period;
    }
    return m < dim ? m : period - m;
}

__kernel void pad_f32(__global const float* x, __global float* y, __global const long* meta, long rank, int mode,
                      float fill, long count) {
    long i = get_global_id(0);
    if (i < count) {
        long remaining = i;
        long offset = 0;
        int filled = 0;
        for (long j = 0; j < rank; ++j) {
            long stride = meta[48 + j];
            long coord = remaining / stride;
            remaining -= coord * stride;
            long src = MapPadPosition(coord - meta[j], meta[16 + j], mode);
            if (src < 0) {
                filled = 1;
                break;
            }
            offset += src * meta[32 + j];
        }
        y[i] = filled ? fill : x[offset];
    }
}
)CLC";

constexpr const char* kPadInt64Source = R"CLC(
long MapPadPosition(long pos, long dim, int mode) {
    if (pos >= 0 && pos < dim) {
        return pos;
    }
    if (mode == 0) {
        return -1;
    }
    if (mode == 2) {
        return pos < 0 ? 0 : dim - 1;
    }
    if (dim == 1) {
        return 0;
    }
    long period = 2 * (dim - 1);
    long m = pos % period;
    if (m < 0) {
        m += period;
    }
    return m < dim ? m : period - m;
}

__kernel void pad_i64(__global const long* x, __global long* y, __global const long* meta, long rank, int mode,
                      long fill, long count) {
    long i = get_global_id(0);
    if (i < count) {
        long remaining = i;
        long offset = 0;
        int filled = 0;
        for (long j = 0; j < rank; ++j) {
            long stride = meta[48 + j];
            long coord = remaining / stride;
            remaining -= coord * stride;
            long src = MapPadPosition(coord - meta[j], meta[16 + j], mode);
            if (src < 0) {
                filled = 1;
                break;
            }
            offset += src * meta[32 + j];
        }
        y[i] = filled ? fill : x[offset];
    }
}
)CLC";

} // namespace

void InstanceNormalization(const Node& node, const std::vector<TensorView>& inputs,
                           const std::vector<TensorView>& outputs) {
    const std::string context = "InstanceNormalization (node '" + node.name + "')";
    if (inputs.size() != 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected input, scale, B, and one output");
    }
    const TensorView& x = inputs[0];
    const TensorView& y = outputs[0];
    if (x.dtype != DataType::Float32 || y.dtype != DataType::Float32) {
        throw InferenceException(context + ": input and output must be float32");
    }
    if (x.dims.size() < 2 || x.dims != y.dims) {
        throw InferenceException(context + ": output must have the shape of an input of rank >= 2");
    }
    if (x.dims.size() > kMaxRank) {
        throw InferenceException(context + ": input rank above the supported maximum (16)");
    }
    for (int64_t dim : x.dims) {
        if (dim <= 0) {
            throw InferenceException(context + ": non-positive dimension");
        }
    }
    if (x.data == nullptr || y.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    const int64_t batch = x.dims[0];
    const int64_t channels = x.dims[1];
    int64_t spatial = 1;
    for (std::size_t i = 2; i < x.dims.size(); ++i) {
        spatial *= x.dims[i];
    }
    const cl_mem scale = ChannelParam(inputs[1], channels, context, "scale");
    const cl_mem bias = ChannelParam(inputs[2], channels, context, "B");
    const double epsilon = static_cast<double>(FloatAttribute(node, "epsilon", 1e-5F));
    const int64_t rows = batch * channels;
    if (rows == 0) {
        return;
    }
    // The kernel accumulates in double; OpenCL fp64 is optional, so fail
    // with the device capability named instead of a build error.
    cl_device_id device = nullptr;
    CheckCl(clGetContextInfo(opencl_runtime::Context(), CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr),
            "clGetContextInfo", context);
    cl_device_fp_config fp64 = 0;
    CheckCl(clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp64), &fp64, nullptr), "clGetDeviceInfo",
            context);
    if (fp64 == 0) {
        throw InferenceException(context + ": device has no double-precision support (cl_khr_fp64 is required)");
    }
    cl_mem xm = reinterpret_cast<cl_mem>(x.data);
    cl_mem ym = reinterpret_cast<cl_mem>(y.data);
    cl_long rowsArg = static_cast<cl_long>(rows);
    cl_long channelsArg = static_cast<cl_long>(channels);
    cl_long spatialArg = static_cast<cl_long>(spatial);
    cl_double epsilonArg = static_cast<cl_double>(epsilon);
    cl_kernel kernel = opencl_runtime::KernelFor("InstanceNormalization", kInstnormSource, "instnorm_f");
    CheckCl(clSetKernelArg(kernel, 0, sizeof(xm), &xm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(scale), &scale), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(bias), &bias), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(rowsArg), &rowsArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(channelsArg), &channelsArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(spatialArg), &spatialArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 7, sizeof(epsilonArg), &epsilonArg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(rows);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

void Range(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Range (node '" + node.name + "')";
    if (inputs.size() != 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected start, limit, delta and one output");
    }
    const TensorView& out = outputs[0];
    const DataType dtype = inputs[0].dtype;
    if ((dtype != DataType::Float32 && dtype != DataType::Int64) || out.dtype != dtype) {
        throw InferenceException(context + ": tensors must share dtype float32 or int64");
    }
    if (out.dims.size() != 1 || out.dims[0] < 0) {
        throw InferenceException(context + ": output must be rank 1");
    }
    const int64_t outCount = out.dims[0];

    if (dtype == DataType::Float32) {
        const float start = ReadDeviceScalar<float>(inputs[0], dtype, "start", context);
        const float limit = ReadDeviceScalar<float>(inputs[1], dtype, "limit", context);
        const float delta = ReadDeviceScalar<float>(inputs[2], dtype, "delta", context);
        if (delta == 0.0F) {
            throw InferenceException(context + ": delta must be non-zero");
        }
        const double span =
            std::ceil((static_cast<double>(limit) - static_cast<double>(start)) / static_cast<double>(delta));
        const int64_t expected = span > 0.0 ? static_cast<int64_t>(span) : 0;
        if (outCount != expected) {
            throw InferenceException(context + ": output length does not match the range");
        }
        if (outCount == 0) {
            return;
        }
        if (out.data == nullptr) {
            throw InferenceException(context + ": null output buffer");
        }
        cl_mem ym = reinterpret_cast<cl_mem>(out.data);
        cl_float startArg = static_cast<cl_float>(start);
        cl_float deltaArg = static_cast<cl_float>(delta);
        cl_long countArg = static_cast<cl_long>(outCount);
        cl_kernel kernel = opencl_runtime::KernelFor("Range/float", kRangeFloatSource, "range_f32");
        CheckCl(clSetKernelArg(kernel, 0, sizeof(ym), &ym), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 1, sizeof(startArg), &startArg), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 2, sizeof(deltaArg), &deltaArg), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 3, sizeof(countArg), &countArg), "clSetKernelArg", context);
        const std::size_t global = static_cast<std::size_t>(outCount);
        CheckCl(
            clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
        CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
        return;
    }

    const int64_t start = ReadDeviceScalar<int64_t>(inputs[0], dtype, "start", context);
    const int64_t limit = ReadDeviceScalar<int64_t>(inputs[1], dtype, "limit", context);
    const int64_t delta = ReadDeviceScalar<int64_t>(inputs[2], dtype, "delta", context);
    if (delta == 0) {
        throw InferenceException(context + ": delta must be non-zero");
    }
    int64_t expected = 0;
    if (delta > 0 && limit > start) {
        expected = (limit - start + delta - 1) / delta;
    } else if (delta < 0 && limit < start) {
        expected = (start - limit + (-delta) - 1) / (-delta);
    }
    if (outCount != expected) {
        throw InferenceException(context + ": output length does not match the range");
    }
    if (outCount == 0) {
        return;
    }
    if (out.data == nullptr) {
        throw InferenceException(context + ": null output buffer");
    }
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    cl_long startArg = static_cast<cl_long>(start);
    cl_long deltaArg = static_cast<cl_long>(delta);
    cl_long countArg = static_cast<cl_long>(outCount);
    cl_kernel kernel = opencl_runtime::KernelFor("Range/int64", kRangeInt64Source, "range_i64");
    CheckCl(clSetKernelArg(kernel, 0, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(startArg), &startArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(deltaArg), &deltaArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(countArg), &countArg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(outCount);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

void Pad(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Pad (node '" + node.name + "')";
    if (inputs.size() < 2 || inputs.size() > 4 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two to four inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if ((in.dtype != DataType::Float32 && in.dtype != DataType::Int64) || out.dtype != in.dtype) {
        throw InferenceException(context + ": data tensors must share dtype float32 or int64");
    }

    int mode = 0; // the kernel's encoding: 0 constant, 1 reflect, 2 edge
    for (const Attribute& attr : node.attributes) {
        if (attr.name != "mode") {
            continue;
        }
        const std::string* value = std::get_if<std::string>(&attr.value);
        if (value == nullptr) {
            throw InferenceException(context + ": attribute 'mode' must be a string");
        }
        if (*value == "constant") {
            mode = 0;
        } else if (*value == "reflect") {
            mode = 1;
        } else if (*value == "edge") {
            mode = 2;
        } else {
            throw InferenceException(context + ": unsupported mode '" + *value + "'");
        }
    }

    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (rank > static_cast<int64_t>(kMaxRank)) {
        throw InferenceException(context + ": input rank above the supported maximum (16)");
    }
    (void)ElementCount(in.dims, context);
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const std::vector<int64_t> pads = ReadDeviceInt64s(inputs[1], "pads", context);
    std::vector<int64_t> axes;
    if (inputs.size() == 4 && inputs[3].dtype != DataType::Unknown) {
        axes = ReadDeviceInt64s(inputs[3], "axes", context);
    } else {
        for (int64_t i = 0; i < rank; ++i) {
            axes.push_back(i);
        }
    }
    if (pads.size() != axes.size() * 2) {
        throw InferenceException(context + ": pads must hold two entries per padded axis");
    }

    std::vector<int64_t> begins(static_cast<std::size_t>(rank), 0);
    std::vector<int64_t> ends(static_cast<std::size_t>(rank), 0);
    std::vector<bool> seen(static_cast<std::size_t>(rank), false);
    for (std::size_t k = 0; k < axes.size(); ++k) {
        int64_t axis = axes[k];
        if (axis < -rank || axis >= rank) {
            throw InferenceException(context + ": axis out of range");
        }
        if (axis < 0) {
            axis += rank;
        }
        if (seen[static_cast<std::size_t>(axis)]) {
            throw InferenceException(context + ": duplicate axis");
        }
        seen[static_cast<std::size_t>(axis)] = true;
        begins[static_cast<std::size_t>(axis)] = pads[k];
        ends[static_cast<std::size_t>(axis)] = pads[axes.size() + k];
    }

    std::vector<int64_t> expectedDims = in.dims;
    for (std::size_t i = 0; i < expectedDims.size(); ++i) {
        expectedDims[i] += begins[i] + ends[i];
    }
    if (out.dims != expectedDims) {
        throw InferenceException(context + ": output shape does not match the pads");
    }
    const int64_t outCount = ElementCount(out.dims, context);

    float fillFloat = 0.0F;
    int64_t fillInt = 0;
    if (inputs.size() >= 3 && inputs[2].dtype != DataType::Unknown) {
        if (inputs[2].dtype != in.dtype) {
            throw InferenceException(context + ": constant_value must match the data dtype");
        }
        if (ElementCount(inputs[2].dims, context) != 1 || inputs[2].data == nullptr) {
            throw InferenceException(context + ": constant_value must be a scalar");
        }
        if (in.dtype == DataType::Float32) {
            fillFloat = ReadDeviceScalar<float>(inputs[2], in.dtype, "constant_value", context);
        } else {
            fillInt = ReadDeviceScalar<int64_t>(inputs[2], in.dtype, "constant_value", context);
        }
    }
    if (outCount == 0) {
        return;
    }

    std::vector<int64_t> inStrides(in.dims.size(), 1);
    for (std::size_t i = inStrides.size(); i-- > 1;) {
        inStrides[i - 1] = inStrides[i] * in.dims[i];
    }
    std::vector<int64_t> outStrides(out.dims.size(), 1);
    for (std::size_t i = outStrides.size(); i-- > 1;) {
        outStrides[i - 1] = outStrides[i] * out.dims[i];
    }
    // Kernel metadata layout: see the comment above kPadFloatSource.
    std::vector<cl_long> meta(4 * kMaxRank, 0);
    for (int64_t j = 0; j < rank; ++j) {
        const std::size_t axis = static_cast<std::size_t>(j);
        meta[axis] = static_cast<cl_long>(begins[axis]);
        meta[kMaxRank + axis] = static_cast<cl_long>(in.dims[axis]);
        meta[2 * kMaxRank + axis] = static_cast<cl_long>(inStrides[axis]);
        meta[3 * kMaxRank + axis] = static_cast<cl_long>(outStrides[axis]);
    }
    cl_int err = CL_SUCCESS;
    cl_mem metaBuf =
        clCreateBuffer(opencl_runtime::Context(), CL_MEM_READ_ONLY, sizeof(cl_long) * meta.size(), nullptr, &err);
    CheckCl(err, "clCreateBuffer (pad metadata)", context);
    CheckCl(clEnqueueWriteBuffer(opencl_runtime::Queue(), metaBuf, CL_TRUE, 0, sizeof(cl_long) * meta.size(),
                                 meta.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer (pad metadata)", context);
    const bool isFloat = in.dtype == DataType::Float32;
    cl_kernel kernel =
        opencl_runtime::KernelFor(isFloat ? "Pad/float" : "Pad/int64", isFloat ? kPadFloatSource : kPadInt64Source,
                                  isFloat ? "pad_f32" : "pad_i64");
    cl_mem im = reinterpret_cast<cl_mem>(in.data);
    cl_mem om = reinterpret_cast<cl_mem>(out.data);
    cl_long rankArg = static_cast<cl_long>(rank);
    cl_int modeArg = static_cast<cl_int>(mode);
    cl_long countArg = static_cast<cl_long>(outCount);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(im), &im), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(om), &om), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(metaBuf), &metaBuf), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(rankArg), &rankArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(modeArg), &modeArg), "clSetKernelArg", context);
    if (isFloat) {
        cl_float fillArg = static_cast<cl_float>(fillFloat);
        CheckCl(clSetKernelArg(kernel, 5, sizeof(fillArg), &fillArg), "clSetKernelArg", context);
    } else {
        cl_long fillArg = static_cast<cl_long>(fillInt);
        CheckCl(clSetKernelArg(kernel, 5, sizeof(fillArg), &fillArg), "clSetKernelArg", context);
    }
    CheckCl(clSetKernelArg(kernel, 6, sizeof(countArg), &countArg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(outCount);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
    CheckCl(clReleaseMemObject(metaBuf), "clReleaseMemObject (pad metadata)", context);
}

} // namespace opencl_kernels
} // namespace nert
