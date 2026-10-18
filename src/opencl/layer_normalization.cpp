// LayerNormalization on OpenCL: normalize over the trailing axes from
// `axis` (default -1), then scale and shift.
//
// Mirrors the CPU oracle: one work-item per prod(dims[:axis]) row, two-pass
// float accumulation (mean, then variance) with one rounding at the store,
// Scale/B per normalized element or scalar, optional Mean and InvStdDev
// outputs with one value per row. The OpenCL C source compiles once per
// process via the shared runtime.

#include "nert/device.hpp"
#include "runtime.hpp"

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

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("LayerNormalization: attribute '" + name + "' must be an integer (node '" +
                                         node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

float FloatAttribute(const Node& node, const std::string& name, float fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const float* value = std::get_if<float>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("LayerNormalization: attribute '" + name + "' must be a float (node '" +
                                         node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

// A per-normalized-element parameter (Scale or B): `inner` values, or one.
const float* ParamData(const TensorView& view, int64_t inner, bool& scalar, const std::string& context,
                       const char* what) {
    if (view.dtype != DataType::Float32 || view.data == nullptr) {
        throw InferenceException(context + ": " + what + " must be a float32 tensor");
    }
    const int64_t count = ElementCount(view.dims, context);
    if (count != inner && count != 1) {
        throw InferenceException(context + ": " + what + " must have one value per normalized element");
    }
    scalar = count == 1;
    return static_cast<const float*>(view.data);
}

constexpr const char* kSource = R"CLC(
__kernel void layer_norm(__global const float* x, __global float* y, __global float* meanOut,
                         __global float* invStdOut, __global const float* scale, __global const float* bias,
                         long rows, long inner, float epsilon, int scaleScalar, int hasBias, int biasScalar,
                         int hasMean, int hasInvStd) {
    long r = get_global_id(0);
    if (r < rows) {
        __global const float* row = x + r * inner;
        float sum = 0.0f;
        for (long k = 0; k < inner; ++k) {
            sum += row[k];
        }
        const float mean = sum / (float)inner;
        float squares = 0.0f;
        for (long k = 0; k < inner; ++k) {
            const float centered = row[k] - mean;
            squares += centered * centered;
        }
        const float invStd = 1.0f / sqrt(squares / (float)inner + epsilon);
        __global float* outRow = y + r * inner;
        for (long k = 0; k < inner; ++k) {
            const float normalized = (row[k] - mean) * invStd;
            const float scaled = normalized * scale[scaleScalar != 0 ? 0 : k];
            outRow[k] = hasBias != 0 ? scaled + bias[biasScalar != 0 ? 0 : k] : scaled;
        }
        if (hasMean != 0) {
            meanOut[r] = mean;
        }
        if (hasInvStd != 0) {
            invStdOut[r] = invStd;
        }
    }
}
)CLC";

template <typename T> void SetArg(cl_kernel kernel, int index, T value, const std::string& context) {
    CheckCl(clSetKernelArg(kernel, static_cast<cl_uint>(index), sizeof(T), &value), "clSetKernelArg", context);
}

} // namespace

void LayerNormalization(const Node& node, const std::vector<TensorView>& inputs,
                        const std::vector<TensorView>& outputs) {
    const std::string context = "LayerNormalization (node '" + node.name + "')";
    if (inputs.size() < 2 || inputs.size() > 3 || outputs.empty() || outputs.size() > 3) {
        throw InferenceException(context + ": expected X, Scale, optional B, and one to three outputs");
    }
    const TensorView& x = inputs[0];
    const TensorView& y = outputs[0];
    if (x.dtype != DataType::Float32 || y.dtype != DataType::Float32) {
        throw InferenceException(context + ": X and Y must be float32");
    }
    if (x.dims.empty() || x.dims != y.dims) {
        throw InferenceException(context + ": Y must have the shape of a non-scalar X");
    }
    const int64_t rank = static_cast<int64_t>(x.dims.size());
    int64_t axis = IntAttribute(node, "axis", -1);
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        throw InferenceException(context + ": axis is out of range");
    }
    const float epsilon = FloatAttribute(node, "epsilon", 1e-5F);

    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t i = 0; i < rank; ++i) {
        (i < axis ? outer : inner) *= x.dims[static_cast<std::size_t>(i)];
    }
    (void)ElementCount(x.dims, context);
    if (x.data == nullptr || y.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    bool scaleScalar = false;
    const float* scale = ParamData(inputs[1], inner, scaleScalar, context, "Scale");
    bool biasScalar = false;
    const float* bias = nullptr;
    if (inputs.size() == 3 && inputs[2].dtype != DataType::Unknown) {
        bias = ParamData(inputs[2], inner, biasScalar, context, "B");
    }

    float* meanOut = nullptr;
    float* invStdOut = nullptr;
    for (std::size_t o = 1; o < outputs.size(); ++o) {
        if (outputs[o].dtype == DataType::Unknown) {
            continue;
        }
        if (outputs[o].dtype != DataType::Float32 || outputs[o].data == nullptr ||
            ElementCount(outputs[o].dims, context) != outer) {
            throw InferenceException(context + ": Mean and InvStdDev must be float32 with one value per row");
        }
        (o == 1 ? meanOut : invStdOut) = static_cast<float*>(outputs[o].data);
    }

    if (outer == 0) {
        return;
    }
    cl_mem xm = reinterpret_cast<cl_mem>(const_cast<void*>(x.data));
    cl_mem ym = reinterpret_cast<cl_mem>(y.data);
    // Absent parameters and statistics bind a live buffer that the kernel
    // never touches through its guard flag.
    cl_mem scalem = reinterpret_cast<cl_mem>(const_cast<float*>(scale));
    cl_mem biasm = reinterpret_cast<cl_mem>(const_cast<float*>(bias != nullptr ? bias : scale));
    cl_mem meanm = meanOut != nullptr ? reinterpret_cast<cl_mem>(meanOut) : ym;
    cl_mem invstdm = invStdOut != nullptr ? reinterpret_cast<cl_mem>(invStdOut) : ym;
    cl_long rowsArg = static_cast<cl_long>(outer);
    cl_long innerArg = static_cast<cl_long>(inner);
    cl_int scaleScalarArg = scaleScalar ? 1 : 0;
    cl_int hasBiasArg = bias != nullptr ? 1 : 0;
    cl_int biasScalarArg = biasScalar ? 1 : 0;
    cl_int hasMeanArg = meanOut != nullptr ? 1 : 0;
    cl_int hasInvStdArg = invStdOut != nullptr ? 1 : 0;
    cl_kernel kernel = opencl_runtime::KernelFor("LayerNormalization", kSource, "layer_norm");
    SetArg(kernel, 0, xm, context);
    SetArg(kernel, 1, ym, context);
    SetArg(kernel, 2, meanm, context);
    SetArg(kernel, 3, invstdm, context);
    SetArg(kernel, 4, scalem, context);
    SetArg(kernel, 5, biasm, context);
    SetArg(kernel, 6, rowsArg, context);
    SetArg(kernel, 7, innerArg, context);
    SetArg(kernel, 8, epsilon, context);
    SetArg(kernel, 9, scaleScalarArg, context);
    SetArg(kernel, 10, hasBiasArg, context);
    SetArg(kernel, 11, biasScalarArg, context);
    SetArg(kernel, 12, hasMeanArg, context);
    SetArg(kernel, 13, hasInvStdArg, context);
    const std::size_t global = static_cast<std::size_t>(outer);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
