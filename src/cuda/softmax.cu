// Softmax on CUDA: numerically stable softmax along an axis, one thread per
// (outer, inner) row.
//
// Mirrors the CPU oracle: float32, output shape equals input shape, axis
// wraps negatives, max-subtracted exponentials accumulated in double with one
// float32 rounding per output. The thread layout (one row per thread) keeps
// the accumulation order identical to the oracle, so results match bit for
// bit on the same inputs.

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

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback, const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute '" + name + "' must be an integer");
            }
            return *value;
        }
    }
    return fallback;
}

__global__ void SoftmaxKernel(const float* x, float* y, int64_t rows, int64_t dim, int64_t inner) {
    for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += blockDim.x * gridDim.x) {
        const int64_t o = inner == 0 ? row : row / inner;
        const int64_t n = inner == 0 ? 0 : row % inner;
        float peak = x[(o * dim) * inner + n];
        for (int64_t k = 1; k < dim; ++k) {
            const float value = x[(o * dim + k) * inner + n];
            if (value > peak) {
                peak = value;
            }
        }
        double total = 0.0;
        for (int64_t k = 0; k < dim; ++k) {
            total += exp(static_cast<double>(x[(o * dim + k) * inner + n]) - static_cast<double>(peak));
        }
        for (int64_t k = 0; k < dim; ++k) {
            const double shifted = exp(static_cast<double>(x[(o * dim + k) * inner + n]) - static_cast<double>(peak));
            y[(o * dim + k) * inner + n] = static_cast<float>(shifted / total);
        }
    }
}

} // namespace

void Softmax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Softmax (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (in.dims != out.dims) {
        throw InferenceException(context + ": output shape must match the input shape");
    }
    (void)ElementCount(out.dims, context);
    (void)ElementCount(in.dims, context);
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const int64_t rank = static_cast<int64_t>(in.dims.size());
    int64_t axis = IntAttribute(node, "axis", -1, context);
    if (axis < -rank || axis >= rank) {
        throw InferenceException(context + ": axis out of range");
    }
    if (axis < 0) {
        axis += rank;
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= in.dims[static_cast<std::size_t>(i)];
    }
    const int64_t dim = rank == 0 ? 1 : in.dims[static_cast<std::size_t>(axis)];
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= in.dims[static_cast<std::size_t>(i)];
    }
    const int64_t rows = outer * inner;
    if (rows == 0) {
        return;
    }
    const int block = 256;
    const int grid = static_cast<int>((rows + block - 1) / block);
    SoftmaxKernel<<<grid, block>>>(static_cast<const float*>(in.data), static_cast<float*>(out.data), rows, dim, inner);
    CheckCuda(cudaGetLastError(), "SoftmaxKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace cuda_kernels
} // namespace nert
