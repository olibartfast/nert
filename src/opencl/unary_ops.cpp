// Unary math rows on OpenCL: IsNaN, Erf, Elu.
//
// Mirrors the CPU oracles: float32 elementwise maps, bool output for IsNaN,
// `alpha` (default 1.0) for Elu. One work-item per element.

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

float FloatAttribute(const Node& node, const std::string& name, float fallback, const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const float* value = std::get_if<float>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute '" + name + "' must be a float");
            }
            return *value;
        }
    }
    return fallback;
}

constexpr const char* kSource = R"CLC(
__kernel void is_nan(__global const float* x, __global uchar* y) {
    y[get_global_id(0)] = isnan(x[get_global_id(0)]) ? 1 : 0;
}
__kernel void erf_f(__global const float* x, __global float* y) {
    y[get_global_id(0)] = erf(x[get_global_id(0)]);
}
__kernel void elu_f(__global const float* x, __global float* y, float alpha) {
    long i = get_global_id(0);
    float v = x[i];
    y[i] = v >= 0.0f ? v : alpha * (exp(v) - 1.0f);
}
)CLC";

void Launch1(const std::string& context, cl_kernel kernel, cl_mem x, cl_mem y, int64_t count) {
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(y), &y), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace

void IsNaN(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "IsNaN (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Bool) {
        throw InferenceException(context + ": input must be float32 and output bool");
    }
    const int64_t count = ElementCount(in.dims, context);
    if (ElementCount(out.dims, context) != count) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    Launch1(context, opencl_runtime::KernelFor("IsNaN", kSource, "is_nan"), x, y, count);
}

void Erf(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Erf (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    const int64_t count = ElementCount(in.dims, context);
    if (ElementCount(out.dims, context) != count) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    Launch1(context, opencl_runtime::KernelFor("Erf", kSource, "erf_f"), x, y, count);
}

void Elu(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Elu (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    const int64_t count = ElementCount(in.dims, context);
    if (ElementCount(out.dims, context) != count) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    const float alpha = FloatAttribute(node, "alpha", 1.0F, context);
    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    cl_kernel kernel = opencl_runtime::KernelFor("Elu", kSource, "elu_f");
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(alpha), &alpha), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
