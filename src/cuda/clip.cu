// Clip on CUDA: elementwise clamp to optional scalar min/max inputs.
//
// Mirrors the CPU oracle (opset-18 inputs, not attributes): float32 data,
// min/max scalar float32 inputs, Unknown-dtype = absent with infinite
// fallbacks. Scalar bounds are downloaded to the host; one thread per
// element clamps.

#include "nert/device.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <string>
#include <vector>

namespace nert {
namespace cuda_kernels {
namespace {

void CheckCuda(cudaError_t status, const char* what, const std::string& context) {
    if (status != cudaSuccess) {
        throw InferenceException(context + ": " + what + " failed: " + cudaGetErrorString(status));
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

// Reads an optional scalar bound from the device; returns `fallback` when
// the input is omitted (Unknown dtype).
float ReadBound(const TensorView& view, float fallback, const std::string& context) {
    if (view.dtype == DataType::Unknown) {
        return fallback;
    }
    if (view.dtype != DataType::Float32) {
        throw InferenceException(context + ": min/max must be float32");
    }
    if (ElementCount(view.dims, context) != 1) {
        throw InferenceException(context + ": min/max must be scalar");
    }
    if (view.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    float value = 0.0F;
    CheckCuda(cudaMemcpy(&value, view.data, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy", context);
    return value;
}

__global__ void ClipKernel(const float* x, float* y, int64_t count, float lo, float hi) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        float v = x[i] < lo ? lo : x[i];
        y[i] = v > hi ? hi : v;
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
    static const TensorView kAbsent{};
    const float lo =
        ReadBound(inputs.size() > 1 ? inputs[1] : kAbsent, -std::numeric_limits<float>::infinity(), context);
    const float hi =
        ReadBound(inputs.size() > 2 ? inputs[2] : kAbsent, std::numeric_limits<float>::infinity(), context);

    if (count == 0) {
        return;
    }
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    ClipKernel<<<grid, block>>>(static_cast<const float*>(in.data), static_cast<float*>(out.data), count, lo, hi);
    CheckCuda(cudaGetLastError(), "ClipKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace cuda_kernels
} // namespace nert
