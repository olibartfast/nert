// Hipified from src/cuda/squeeze.cu via hipify-perl, hand-reviewed.
// Squeeze on HIP: drop size-1 axes by copying the buffer.
//
// Mirrors the CPU oracle: the element order is unchanged, so the kernel
// copies the buffer and shape inference owns the output dims. Float32,
// int64, or bool data with matching dtype and element count.

#include "nert/device.hpp"

#include <cstddef>
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

} // namespace

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
    if (in.data != out.data) {
        CheckHip(hipMemcpy(out.data, in.data, static_cast<std::size_t>(count) * element, hipMemcpyDeviceToDevice),
                 "hipMemcpy", context);
    }
}

} // namespace hip_kernels
} // namespace nert
