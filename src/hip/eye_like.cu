// Hipified from src/cuda/eye_like.cu via hipify-perl, hand-reviewed.
// EyeLike on HIP: 2-D output with the input shape, ones on the k-th
// diagonal and zeros elsewhere.
//
// Mirrors the CPU oracle: shared rank-2 shape, output dtype float32,
// int64, or bool (fixed by shape inference), `k` int attr default 0.
// Zero-fill via memset, then one thread per row writes the diagonal one.

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

template <typename T> __global__ void EyeDiagonalKernel(T* y, int64_t rows, int64_t cols, int64_t k, T one) {
    for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
        const int64_t c = r + k;
        if (c >= 0 && c < cols) {
            y[r * cols + c] = one;
        }
    }
}

template <typename T>
void LaunchEye(T* y, int64_t rows, int64_t cols, int64_t k, T one, std::size_t bytes, const std::string& context) {
    CheckHip(hipMemset(y, 0, bytes), "hipMemset", context);
    if (rows == 0) {
        return;
    }
    const int block = 256;
    const int grid = static_cast<int>((rows + block - 1) / block);
    EyeDiagonalKernel<<<grid, block>>>(y, rows, cols, k, one);
    CheckHip(hipGetLastError(), "EyeDiagonalKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace

void EyeLike(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "EyeLike (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dims.size() != 2 || out.dims != in.dims) {
        throw InferenceException(context + ": input and output must share a rank-2 shape");
    }
    if (out.dtype != DataType::Float32 && out.dtype != DataType::Int64 && out.dtype != DataType::Bool) {
        throw InferenceException(context + ": output must be float32, int64, or bool");
    }
    const int64_t k = IntAttribute(node, "k", 0, context);
    const int64_t rows = in.dims[0];
    const int64_t cols = in.dims[1];
    const int64_t count = ElementCount(out.dims, context);
    if (out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (count == 0) {
        return;
    }
    if (out.dtype == DataType::Float32) {
        LaunchEye(static_cast<float*>(out.data), rows, cols, k, 1.0F, static_cast<std::size_t>(count) * sizeof(float),
                  context);
    } else if (out.dtype == DataType::Int64) {
        LaunchEye(static_cast<int64_t*>(out.data), rows, cols, k, int64_t{1},
                  static_cast<std::size_t>(count) * sizeof(int64_t), context);
    } else {
        LaunchEye(static_cast<bool*>(out.data), rows, cols, k, true, static_cast<std::size_t>(count) * sizeof(bool),
                  context);
    }
}

} // namespace hip_kernels
} // namespace nert
