// Not on CUDA: elementwise logical negation, one thread per element.
//
// Mirrors the CPU oracle: bool input and output with matching shapes.

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

__global__ void NotKernel(const bool* x, bool* y, int64_t count) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        y[i] = !x[i];
    }
}

} // namespace

void Not(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Not (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Bool || out.dtype != DataType::Bool) {
        throw InferenceException(context + ": input and output must be bool");
    }
    if (out.dims != in.dims) {
        throw InferenceException(context + ": output shape must match the input shape");
    }
    const int64_t count = ElementCount(out.dims, context);
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (count == 0) {
        return;
    }
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    NotKernel<<<grid, block>>>(static_cast<const bool*>(in.data), static_cast<bool*>(out.data), count);
    CheckCuda(cudaGetLastError(), "NotKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace cuda_kernels
} // namespace nert
