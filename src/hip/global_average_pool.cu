// Hipified from src/cuda/global_average_pool.cu via hipify-perl, hand-reviewed.
// GlobalAveragePool on HIP: whole-spatial mean, one thread per (n, c) row.
//
// Mirrors the CPU oracle's validation; data is float32, the input has batch
// and channel axes, and the output is the input shape with all spatial axes
// at 1. Each row accumulates in double with one float32 rounding, in the
// same order as the oracle, so results match bit for bit.

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

__global__ void GlobalAveragePoolKernel(const float* x, float* y, int64_t rows, int64_t spatial) {
    for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += blockDim.x * gridDim.x) {
        double total = 0.0;
        for (int64_t k = 0; k < spatial; ++k) {
            total += static_cast<double>(x[row * spatial + k]);
        }
        y[row] = static_cast<float>(total / static_cast<double>(spatial));
    }
}

} // namespace

void GlobalAveragePool(const Node& node, const std::vector<TensorView>& inputs,
                       const std::vector<TensorView>& outputs) {
    const std::string context = "GlobalAveragePool (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (in.dims.size() < 2) {
        throw InferenceException(context + ": input must have batch and channel axes (rank at least 2)");
    }
    (void)ElementCount(in.dims, context);
    (void)ElementCount(out.dims, context);
    std::vector<int64_t> expected = {in.dims[0], in.dims[1]};
    expected.insert(expected.end(), in.dims.size() - 2, 1);
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape must be the input shape with all spatial axes at 1");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    const int64_t rows = in.dims[0] * in.dims[1];
    int64_t spatial = 1;
    for (std::size_t i = 2; i < in.dims.size(); ++i) {
        spatial *= in.dims[i];
    }
    if (rows == 0) {
        return;
    }
    const int block = 256;
    const int grid = static_cast<int>((rows + block - 1) / block);
    GlobalAveragePoolKernel<<<grid, block>>>(static_cast<const float*>(in.data), static_cast<float*>(out.data), rows,
                                             spatial);
    CheckHip(hipGetLastError(), "GlobalAveragePoolKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace hip_kernels
} // namespace nert
