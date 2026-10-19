// Clip on CUDA: elementwise clamp to optional scalar min/max inputs.
//
// Mirrors the CPU oracle (opset-18 inputs, not attributes): float32 or int64
// data, min/max scalar inputs of the same dtype, Unknown-dtype = absent with
// identity fallbacks (infinite floats, INT64_MIN/INT64_MAX). Scalar bounds are
// downloaded to the host; one thread per element clamps.

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
// the input is omitted (Unknown dtype). A present bound must carry `dtype`.
template <typename T>
T ReadBound(const TensorView& view, T fallback, DataType dtype, const char* type_name, const std::string& context) {
    if (view.dtype == DataType::Unknown) {
        return fallback;
    }
    if (view.dtype != dtype) {
        throw InferenceException(context + ": min/max must be " + type_name);
    }
    if (ElementCount(view.dims, context) != 1) {
        throw InferenceException(context + ": min/max must be scalar");
    }
    if (view.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    T value = T{0};
    CheckCuda(cudaMemcpy(&value, view.data, sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy", context);
    return value;
}

template <typename T> __global__ void ClipKernel(const T* x, T* y, int64_t count, T lo, T hi) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        T v = x[i] < lo ? lo : x[i];
        y[i] = v > hi ? hi : v;
    }
}

// Validates bounds, downloads them, and launches one thread per element.
template <typename T>
void RunClip(const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs, int64_t count,
             const std::string& context, T lo_fallback, T hi_fallback, DataType dtype, const char* type_name) {
    static const TensorView kAbsent{};
    const T lo = ReadBound(inputs.size() > 1 ? inputs[1] : kAbsent, lo_fallback, dtype, type_name, context);
    const T hi = ReadBound(inputs.size() > 2 ? inputs[2] : kAbsent, hi_fallback, dtype, type_name, context);

    if (count == 0) {
        return;
    }
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    ClipKernel<T>
        <<<grid, block>>>(static_cast<const T*>(inputs[0].data), static_cast<T*>(outputs[0].data), count, lo, hi);
    CheckCuda(cudaGetLastError(), "ClipKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace

void Clip(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Clip (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one to three inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != out.dtype) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be float32");
    }
    const int64_t count = ElementCount(out.dims, context);
    if (ElementCount(in.dims, context) != count) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (in.dtype == DataType::Float32) {
        RunClip<float>(inputs, outputs, count, context, -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::infinity(), DataType::Float32, "float32");
    } else {
        RunClip<int64_t>(inputs, outputs, count, context, std::numeric_limits<int64_t>::min(),
                         std::numeric_limits<int64_t>::max(), DataType::Int64, "int64");
    }
}

} // namespace cuda_kernels
} // namespace nert
