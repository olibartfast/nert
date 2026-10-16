// Gemm on CUDA: Y = alpha * op(A) * op(B) + beta * C via cuBLAS.
//
// Row-major NERT layouts reach column-major cuBLAS as C^T = op(B)^T *
// op(A)^T: the B operand is passed first with leading dim b.dims[1], the A
// operand second with leading dim a.dims[1], and m/n are swapped. TF32 is
// forbidden (CUBLAS_DEFAULT_MATH) so results match the float32 CPU oracle
// within tolerance rather than tensor-float. The optional rank-2 bias rides
// the beta term; a rank-1 [N] bias is added by a small first-party kernel.
// The output is zeroed first so a beta of 0 never reads garbage.

#include "nert/device.hpp"

#include <cstdint>
#include <cublas_v2.h>
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

void CheckCublas(cublasStatus_t status, const char* what, const std::string& context) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw InferenceException(context + ": " + what + " failed (cuBLAS status " +
                                 std::to_string(static_cast<int>(status)) + ")");
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

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("Gemm: attribute '" + name + "' must be an integer (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

float FloatAttribute(const Node& node, const std::string& name, float fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const float* value = std::get_if<float>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("Gemm: attribute '" + name + "' must be a float (node '" + node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

// y[i] += beta * bias[i % n]: the rank-1 [N] bias cuBLAS cannot fold.
__global__ void BiasAddKernel(float* y, const float* bias, float beta, int64_t count, int64_t n) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        y[i] += beta * bias[i % n];
    }
}

} // namespace

void Gemm(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Gemm (node '" + node.name + "')";
    if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two or three inputs and one output");
    }
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView& out = outputs[0];
    const bool hasC = inputs.size() == 3;
    if (a.dtype != DataType::Float32 || b.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (hasC && inputs[2].dtype != DataType::Float32) {
        throw InferenceException(context + ": bias tensor must be float32");
    }
    if (a.dims.size() != 2 || b.dims.size() != 2) {
        throw InferenceException(context + ": A and B must be rank-2");
    }
    (void)ElementCount(a.dims, context);
    (void)ElementCount(b.dims, context);

    const int64_t transA = IntAttribute(node, "transA", 0);
    const int64_t transB = IntAttribute(node, "transB", 0);
    const float alpha = FloatAttribute(node, "alpha", 1.0F);
    const float beta = FloatAttribute(node, "beta", 1.0F);

    const int64_t m = transA == 0 ? a.dims[0] : a.dims[1];
    const int64_t k = transA == 0 ? a.dims[1] : a.dims[0];
    const int64_t bK = transB == 0 ? b.dims[0] : b.dims[1];
    const int64_t n = transB == 0 ? b.dims[1] : b.dims[0];
    if (k != bK) {
        throw InferenceException(context + ": contraction dimension mismatch");
    }
    if (out.dims != std::vector<int64_t>{m, n}) {
        throw InferenceException(context + ": output shape does not match the operands");
    }
    (void)ElementCount(out.dims, context);

    const float* bias = nullptr;
    int64_t biasRank = 0;
    if (hasC) {
        const TensorView& c = inputs[2];
        const bool rankOne = c.dims.size() == 1 && c.dims[0] == n;
        const bool rankTwo = c.dims.size() == 2 && c.dims[0] == m && c.dims[1] == n;
        if (!rankOne && !rankTwo) {
            throw InferenceException(context + ": bias shape must be [N] or [M,N]");
        }
        (void)ElementCount(c.dims, context);
        bias = static_cast<const float*>(c.data);
        biasRank = static_cast<int64_t>(c.dims.size());
    }
    if (a.data == nullptr || b.data == nullptr || out.data == nullptr || (hasC && bias == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    cublasHandle_t handle = nullptr;
    CheckCublas(cublasCreate(&handle), "cublasCreate", context);
    // Forbid TF32/tensor cores: the CPU oracle is strict float32.
    CheckCublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode", context);

    float* y = static_cast<float*>(out.data);
    CheckCuda(cudaMemset(y, 0, static_cast<std::size_t>(m * n) * sizeof(float)), "cudaMemset", context);

    const float betaEff = (hasC && biasRank == 2) ? beta : 0.0F;
    const float* cPtr = (hasC && biasRank == 2) ? bias : y;
    CheckCublas(cublasSgemm(handle, transB == 0 ? CUBLAS_OP_N : CUBLAS_OP_T, transA == 0 ? CUBLAS_OP_N : CUBLAS_OP_T,
                            static_cast<int>(n), static_cast<int>(m), static_cast<int>(k), &alpha,
                            static_cast<const float*>(b.data), static_cast<int>(b.dims[1]),
                            static_cast<const float*>(a.data), static_cast<int>(a.dims[1]), &betaEff, y,
                            static_cast<int>(n)),
                "cublasSgemm", context);

    if (hasC && biasRank == 1 && beta != 0.0F) {
        const int64_t count = m * n;
        const int block = 256;
        const int grid = static_cast<int>((count + block - 1) / block);
        BiasAddKernel<<<grid, block>>>(y, bias, beta, count, n);
        CheckCuda(cudaGetLastError(), "BiasAddKernel launch", context);
    }
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
    CheckCublas(cublasDestroy(handle), "cublasDestroy", context);
}

} // namespace cuda_kernels
} // namespace nert
