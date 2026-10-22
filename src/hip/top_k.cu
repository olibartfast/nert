// Hipified from src/cuda/top_k.cu via hipify-perl, hand-reviewed.
// TopK on HIP: largest-K selection along an axis, one thread per row.
//
// Mirrors the CPU oracle: float32 input, int64 scalar K operand (downloaded
// from device), largest=1/sorted=1 required, negative-wrapped axis, output
// shapes equal to the input with K on the axis. Selection replays the
// oracle's order (value desc, index asc), so values and indices match.

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

__global__ void TopKKernel(const float* x, float* y, int64_t* iy, int64_t rows, int64_t dim, int64_t k, int64_t inner) {
    for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += blockDim.x * gridDim.x) {
        const int64_t o = row / inner;
        const int64_t n = row % inner;
        for (int64_t t = 0; t < k; ++t) {
            int64_t best = -1;
            float bestVal = 0.0F;
            for (int64_t c = 0; c < dim; ++c) {
                bool taken = false;
                for (int64_t u = 0; u < t; ++u) {
                    if (iy[(o * k + u) * inner + n] == c) {
                        taken = true;
                        break;
                    }
                }
                if (taken) {
                    continue;
                }
                const float v = x[(o * dim + c) * inner + n];
                if (best < 0 || v > bestVal) {
                    best = c;
                    bestVal = v;
                }
            }
            y[(o * k + t) * inner + n] = bestVal;
            iy[(o * k + t) * inner + n] = best;
        }
    }
}

} // namespace

void TopK(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "TopK (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 2) {
        throw InferenceException(context + ": expected an input, a K input, and values/indices outputs");
    }
    const TensorView& in = inputs[0];
    const TensorView& k_view = inputs[1];
    const TensorView& values = outputs[0];
    const TensorView& indices = outputs[1];
    if (in.dtype != DataType::Float32) {
        throw InferenceException(context + ": TopK input must be float32");
    }
    if (values.dtype != DataType::Float32) {
        throw InferenceException(context + ": TopK values must be float32");
    }
    if (indices.dtype != DataType::Int64) {
        throw InferenceException(context + ": TopK indices must be int64");
    }
    if (k_view.dtype != DataType::Int64) {
        throw InferenceException(context + ": TopK requires a K input");
    }
    if (IntAttribute(node, "largest", 1, context) != 1) {
        throw InferenceException(context + ": TopK 'largest' must be 1 here");
    }
    if (IntAttribute(node, "sorted", 1, context) != 1) {
        throw InferenceException(context + ": TopK 'sorted' must be 1 here");
    }
    if (k_view.data == nullptr) {
        throw InferenceException(context + ": null K buffer");
    }
    int64_t k = 0;
    CheckHip(hipMemcpy(&k, k_view.data, sizeof(int64_t), hipMemcpyDeviceToHost), "hipMemcpy", context);
    if (k <= 0) {
        throw InferenceException(context + ": TopK K must be positive");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    int64_t axis = IntAttribute(node, "axis", -1, context);
    if (rank == 0 || axis < -rank || axis >= rank) {
        throw InferenceException(context + ": TopK axis is out of range");
    }
    if (axis < 0) {
        axis += rank;
    }
    const int64_t dim = in.dims[static_cast<std::size_t>(axis)];
    if (k > dim) {
        throw InferenceException(context + ": TopK K exceeds the axis dim");
    }
    std::vector<int64_t> expected = in.dims;
    expected[static_cast<std::size_t>(axis)] = k;
    if (values.dims != expected || indices.dims != expected) {
        throw InferenceException(context + ": output shapes do not match the selected K");
    }

    const int64_t values_count = ElementCount(values.dims, context);
    (void)ElementCount(in.dims, context);
    if (values_count > 0 && (in.data == nullptr || values.data == nullptr || indices.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    if (values_count == 0) {
        return;
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= in.dims[static_cast<std::size_t>(i)];
    }
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= in.dims[static_cast<std::size_t>(i)];
    }
    const int64_t rows = outer * inner;
    const int block = 256;
    const int grid = static_cast<int>((rows + block - 1) / block);
    TopKKernel<<<grid, block>>>(static_cast<const float*>(in.data), static_cast<float*>(values.data),
                                static_cast<int64_t*>(indices.data), rows, dim, k, inner);
    CheckHip(hipGetLastError(), "TopKKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace hip_kernels
} // namespace nert
