// Log on CUDA: elementwise natural logarithm, one thread per element.
//
// Mirrors the CPU oracle's validation; data is float32 and the input and
// output counts must match.

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

__global__ void LogKernel(const float* x, float* y, int64_t count) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        y[i] = logf(x[i]);
    }
}

} // namespace

void Log(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Log (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
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
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    LogKernel<<<grid, block>>>(static_cast<const float*>(in.data), static_cast<float*>(out.data), outCount);
    CheckCuda(cudaGetLastError(), "LogKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace cuda_kernels
} // namespace nert
