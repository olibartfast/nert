// Abs on CUDA: elementwise absolute value, one thread per element.
//
// Mirrors the CPU oracle: float32 or int64 sharing one dtype, matching
// element counts.

#include "nert/device.hpp"

#include <cstdint>
#include <cuda_runtime.h>
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

template <typename T> __global__ void AbsKernel(const T* x, T* y, int64_t count) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        y[i] = x[i] >= T{0} ? x[i] : -x[i];
    }
}

template <typename T> void LaunchAbs(const T* x, T* y, int64_t count, const std::string& context) {
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    AbsKernel<<<grid, block>>>(x, y, count);
    CheckCuda(cudaGetLastError(), "AbsKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace

void Abs(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Abs (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if ((in.dtype != DataType::Float32 && in.dtype != DataType::Int64) || out.dtype != in.dtype) {
        throw InferenceException(context + ": tensors must be float32 or int64 with matching dtype");
    }
    const int64_t outCount = ElementCount(out.dims, context);
    const int64_t inCount = ElementCount(in.dims, context);
    if (inCount != outCount) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (outCount == 0) {
        return;
    }
    if (in.dtype == DataType::Float32) {
        LaunchAbs(static_cast<const float*>(in.data), static_cast<float*>(out.data), outCount, context);
    } else {
        LaunchAbs(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), outCount, context);
    }
}

} // namespace cuda_kernels
} // namespace nert
