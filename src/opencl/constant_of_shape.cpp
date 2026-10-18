// ConstantOfShape on OpenCL: fill the output with the scalar `value`.
//
// Mirrors the CPU oracle: one int64 shape input (read back from device
// memory), optional `value` attribute defaulting to float 0, and an output
// dtype selected by the value variant. The fill is clEnqueueFillBuffer, no
// compute kernel.

#include "nert/device.hpp"
#include "runtime.hpp"

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

void CheckCl(cl_int status, const char* what, const std::string& context) {
    if (status != CL_SUCCESS) {
        throw InferenceException(context + ": " + what + " failed (OpenCL status " + std::to_string(status) + ")");
    }
}

int64_t ElementCount(const std::vector<int64_t>& dims, const std::string& context) {
    int64_t count = 1;
    for (int64_t dim : dims) {
        if (dim < 0) {
            throw InferenceException(context + ": negative dimension");
        }
        count *= dim;
    }
    return count;
}

} // namespace

void ConstantOfShape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ConstantOfShape (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected a shape input and one output");
    }
    const TensorView& shape_view = inputs[0];
    const TensorView& out = outputs[0];
    if (shape_view.dtype != DataType::Int64) {
        throw InferenceException(context + ": ConstantOfShape requires a shape input");
    }
    if (shape_view.dims.size() != 1) {
        throw InferenceException(context + ": ConstantOfShape shape input must be rank 1");
    }
    const int64_t extent = ElementCount(shape_view.dims, context);
    if (extent > 0 && shape_view.data == nullptr) {
        throw InferenceException(context + ": null shape buffer");
    }
    std::vector<int64_t> shape(static_cast<std::size_t>(extent), 0);
    if (extent > 0) {
        CheckCl(clEnqueueReadBuffer(
                    opencl_runtime::Queue(), reinterpret_cast<cl_mem>(const_cast<void*>(shape_view.data)), CL_TRUE, 0,
                    static_cast<std::size_t>(extent) * sizeof(int64_t), shape.data(), 0, nullptr, nullptr),
                "clEnqueueReadBuffer", context);
    }
    for (const int64_t v : shape) {
        if (v < 0) {
            throw InferenceException(context + ": ConstantOfShape shape values must be non-negative");
        }
    }

    const Attribute* value = nullptr;
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "value") {
            value = &attr;
        }
    }
    if (out.dims != shape) {
        throw InferenceException(context + ": output shape does not match the shape input");
    }
    int64_t out_count = 1;
    for (const int64_t d : out.dims) {
        out_count *= d;
    }
    if (out_count > 0 && out.data == nullptr) {
        throw InferenceException(context + ": null output buffer");
    }
    if (out_count == 0) {
        return;
    }
    cl_mem y = reinterpret_cast<cl_mem>(out.data);

    if (value == nullptr || std::holds_alternative<float>(value->value)) {
        const float fill = value == nullptr ? 0.0F : std::get<float>(value->value);
        if (out.dtype != DataType::Float32) {
            throw InferenceException(context + ": ConstantOfShape output must be float32 for a float value");
        }
        CheckCl(clEnqueueFillBuffer(opencl_runtime::Queue(), y, &fill, sizeof(fill), 0,
                                    static_cast<std::size_t>(out_count) * sizeof(float), 0, nullptr, nullptr),
                "clEnqueueFillBuffer", context);
    } else if (const int64_t* fill = std::get_if<int64_t>(&value->value)) {
        if (out.dtype != DataType::Int64) {
            throw InferenceException(context + ": ConstantOfShape output must be int64 for an int64 value");
        }
        CheckCl(clEnqueueFillBuffer(opencl_runtime::Queue(), y, fill, sizeof(*fill), 0,
                                    static_cast<std::size_t>(out_count) * sizeof(int64_t), 0, nullptr, nullptr),
                "clEnqueueFillBuffer", context);
    } else if (const bool* fill = std::get_if<bool>(&value->value)) {
        if (out.dtype != DataType::Bool) {
            throw InferenceException(context + ": ConstantOfShape output must be bool for a bool value");
        }
        const std::uint8_t byte = *fill ? 1 : 0;
        CheckCl(clEnqueueFillBuffer(opencl_runtime::Queue(), y, &byte, sizeof(byte), 0,
                                    static_cast<std::size_t>(out_count), 0, nullptr, nullptr),
                "clEnqueueFillBuffer", context);
    } else {
        throw InferenceException(context + ": ConstantOfShape 'value' must be an int64, float, or bool scalar");
    }
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
