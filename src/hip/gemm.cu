// Gemm on HIP (first-party, no library): Y = alpha * op(A) * op(B) + beta * C.
//
// Mirrors the CPU oracle's validation (transposes, alpha/beta, rank-1 [N] or
// rank-2 [M,N] bias) with a naive one-thread-per-output kernel and double
// accumulation, so each output rounds once to float32 like the oracle. No
// rocBLAS/hipBLAS dependency, so the row runs on both the NVIDIA and AMD HIP
// backends.

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

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback, const std::string& op) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(op + ": attribute '" + name + "' must be an integer (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

float FloatAttribute(const Node& node, const std::string& name, float fallback, const std::string& op) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const float* value = std::get_if<float>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(op + ": attribute '" + name + "' must be a float (node '" + node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

struct GemmParams {
    int64_t m;
    int64_t n;
    int64_t k;
    int trans_a;
    int trans_b;
    float alpha;
    float beta;
    int bias_rank; // 0 = absent, 1 = [N], 2 = [M,N]
};

__global__ void GemmKernel(const float* a, const float* b, const float* bias, float* y, int64_t count, GemmParams p) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const int64_t row = i / p.n;
        const int64_t col = i % p.n;
        double acc = 0.0;
        for (int64_t kk = 0; kk < p.k; ++kk) {
            const float av = p.trans_a == 0 ? a[row * p.k + kk] : a[kk * p.m + row];
            const float bv = p.trans_b == 0 ? b[kk * p.n + col] : b[col * p.k + kk];
            acc += static_cast<double>(av) * static_cast<double>(bv);
        }
        double value = static_cast<double>(p.alpha) * acc;
        if (p.bias_rank == 1) {
            value += static_cast<double>(p.beta) * static_cast<double>(bias[col]);
        } else if (p.bias_rank == 2) {
            value += static_cast<double>(p.beta) * static_cast<double>(bias[i]);
        }
        y[i] = static_cast<float>(value);
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

    const int64_t transA = IntAttribute(node, "transA", 0, "Gemm");
    const int64_t transB = IntAttribute(node, "transB", 0, "Gemm");
    const float alpha = FloatAttribute(node, "alpha", 1.0F, "Gemm");
    const float beta = FloatAttribute(node, "beta", 1.0F, "Gemm");

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
    const int64_t outCount = ElementCount(out.dims, context);

    const float* bias = nullptr;
    int biasRank = 0;
    if (hasC) {
        const TensorView& c = inputs[2];
        const bool rankOne = c.dims.size() == 1 && c.dims[0] == n;
        const bool rankTwo = c.dims.size() == 2 && c.dims[0] == m && c.dims[1] == n;
        if (!rankOne && !rankTwo) {
            throw InferenceException(context + ": bias shape must be [N] or [M,N]");
        }
        (void)ElementCount(c.dims, context);
        bias = static_cast<const float*>(c.data);
        biasRank = rankOne ? 1 : 2;
    }
    if (a.data == nullptr || b.data == nullptr || out.data == nullptr || (hasC && bias == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    const GemmParams p{m, n, k, transA == 0 ? 0 : 1, transB == 0 ? 0 : 1, alpha, beta, biasRank};
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    GemmKernel<<<grid, block>>>(static_cast<const float*>(a.data), static_cast<const float*>(b.data), bias,
                                static_cast<float*>(out.data), outCount, p);
    CheckHip(hipGetLastError(), "GemmKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace hip_kernels
} // namespace nert
