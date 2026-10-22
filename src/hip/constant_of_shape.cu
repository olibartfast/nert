// Hipified from src/cuda/constant_of_shape.cu via hipify-perl, hand-reviewed.
// ConstantOfShape on HIP: fill the shape-input dims with one scalar.
//
// Mirrors the CPU oracle: rank-1 int64 shape operand (downloaded from
// device), non-negative values, optional 'value' attr (float/int64/bool
// scalar, default 0.0f) with matching output dtype. One fill kernel.

#include "nert/device.hpp"

#include <cstdint>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

namespace nert {
namespace hip_kernels {
namespace {

void CheckHip(hipError_t status, const char* what, const std::string& context) {
    if (status != hipSuccess) {
        throw InferenceException(context + ": " + what + " failed: " + hipGetErrorString(status));
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

template <typename T> __global__ void FillKernel(T* y, T fill, int64_t count) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        y[i] = fill;
    }
}

template <typename T> void LaunchFill(T* y, T fill, int64_t count, const std::string& context) {
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    FillKernel<<<grid, block>>>(y, fill, count);
    CheckHip(hipGetLastError(), "FillKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
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
        CheckHip(hipMemcpy(shape.data(), shape_view.data, static_cast<std::size_t>(extent) * sizeof(int64_t),
                           hipMemcpyDeviceToHost),
                 "hipMemcpy", context);
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

    if (value == nullptr || std::holds_alternative<float>(value->value)) {
        const float fill = value == nullptr ? 0.0F : std::get<float>(value->value);
        if (out.dtype != DataType::Float32) {
            throw InferenceException(context + ": ConstantOfShape output must be float32 for a float value");
        }
        LaunchFill(static_cast<float*>(out.data), fill, out_count, context);
        return;
    }
    if (const int64_t* fill = std::get_if<int64_t>(&value->value)) {
        if (out.dtype != DataType::Int64) {
            throw InferenceException(context + ": ConstantOfShape output must be int64 for an int64 value");
        }
        LaunchFill(static_cast<int64_t*>(out.data), *fill, out_count, context);
        return;
    }
    if (const bool* fill = std::get_if<bool>(&value->value)) {
        if (out.dtype != DataType::Bool) {
            throw InferenceException(context + ": ConstantOfShape output must be bool for a bool value");
        }
        LaunchFill(static_cast<bool*>(out.data), *fill, out_count, context);
        return;
    }
    throw InferenceException(context + ": ConstantOfShape 'value' must be an int64, float, or bool scalar");
}

} // namespace hip_kernels
} // namespace nert
