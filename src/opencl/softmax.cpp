// Softmax on OpenCL: numerically stable softmax along an axis, one
// work-item per (outer, inner) row.
//
// Mirrors the CPU oracle's validation (float32, output shape equals input
// shape, axis wraps negatives, max-subtracted exponentials). One documented
// deviation from the CUDA row: accumulation is float32 here (OpenCL C
// `double` would need the fp64 extension), so results match the oracle
// within the M2+ tolerance bar, not bit for bit.

#include "nert/device.hpp"
#include "runtime.hpp"

#include <cstdint>
#include <string>
#include <variant>
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
__kernel void softmax(__global const float* x, __global float* y, long rows, long dim, long inner) {
    long row = get_global_id(0);
    if (row < rows) {
        long o = inner == 0 ? row : row / inner;
        long n = inner == 0 ? 0 : row % inner;
        float peak = x[(o * dim) * inner + n];
        for (long k = 1; k < dim; ++k) {
            float value = x[(o * dim + k) * inner + n];
            if (value > peak) {
                peak = value;
            }
        }
        float total = 0.0f;
        for (long k = 0; k < dim; ++k) {
            total += exp(x[(o * dim + k) * inner + n] - peak);
        }
        for (long k = 0; k < dim; ++k) {
            y[(o * dim + k) * inner + n] = exp(x[(o * dim + k) * inner + n] - peak) / total;
        }
    }
}
)CLC";

} // namespace

void Softmax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Softmax (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (in.dims != out.dims) {
        throw InferenceException(context + ": output shape must match the input shape");
    }
    (void)ElementCount(out.dims, context);
    (void)ElementCount(in.dims, context);
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const int64_t rank = static_cast<int64_t>(in.dims.size());
    int64_t axis = IntAttribute(node, "axis", -1, context);
    if (axis < -rank || axis >= rank) {
        throw InferenceException(context + ": axis out of range");
    }
    if (axis < 0) {
        axis += rank;
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= in.dims[static_cast<std::size_t>(i)];
    }
    const int64_t dim = rank == 0 ? 1 : in.dims[static_cast<std::size_t>(axis)];
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= in.dims[static_cast<std::size_t>(i)];
    }
    const int64_t rows = outer * inner;
    if (rows == 0) {
        return;
    }
    cl_kernel kernel = opencl_runtime::KernelFor("Softmax", kSource, "softmax");
    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    cl_long rowsArg = static_cast<cl_long>(rows);
    cl_long dimArg = static_cast<cl_long>(dim);
    cl_long innerArg = static_cast<cl_long>(inner);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(rowsArg), &rowsArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(dimArg), &dimArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(innerArg), &innerArg), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(rows);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
