// Hipified from src/cuda/neg.cu via hipify-perl, hand-reviewed.
// Neg on HIP: elementwise negation, one thread per element.
//
// Mirrors the CPU oracle: float32 or int64 sharing one dtype, matching
// element counts.

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
        if (dim <= 0) {
            throw InferenceException(context + ": non-positive dimension");
        }
        count *= dim;
    }
    return count;
}

template <typename T> __global__ void NegKernel(const T* x, T* y, int64_t count) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        y[i] = -x[i];
    }
}

template <typename T> void LaunchNeg(const T* x, T* y, int64_t count, const std::string& context) {
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    NegKernel<<<grid, block>>>(x, y, count);
    CheckHip(hipGetLastError(), "NegKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace

void Neg(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Neg (node '" + node.name + "')";
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
        LaunchNeg(static_cast<const float*>(in.data), static_cast<float*>(out.data), outCount, context);
    } else {
        LaunchNeg(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), outCount, context);
    }
}

} // namespace hip_kernels
} // namespace nert
