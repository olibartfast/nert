// MatMul on CUDA: batched matrix product via cuBLAS.
//
// Mirrors the CPU oracle: float32 only, 1-D promotion (A [K] -> [1,K],
// B [K] -> [K,1]), NumPy-broadcast batch shapes, per-batch contraction
// check. Each broadcast batch is one cublasSgemm with the Gemm row-major
// swap trick (CUBLAS_PEDANTIC_MATH: strict float32 like the oracle).

#include "nert/device.hpp"

#include <algorithm>
#include <cstddef>
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

// Row-major strides for a contiguous tensor.
std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

// The NumPy multidirectional broadcast of the two batch shapes.
std::vector<int64_t> BroadcastDims(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                                   const std::string& context) {
    const std::size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank, 1);
    for (std::size_t i = 0; i < rank; ++i) {
        const int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
        const int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
        if (da != db && da != 1 && db != 1) {
            throw InferenceException(context + ": batch dimensions are not broadcast-compatible");
        }
        out[i] = std::max(da, db);
    }
    return out;
}

// The batch offset of one input given the broadcast output batch coordinates.
// A missing leading axis and a size-1 axis both broadcast with stride 0.
int64_t BroadcastOffset(const std::vector<int64_t>& inBatch, const std::vector<int64_t>& outBatch,
                        const std::vector<int64_t>& coords) {
    const std::size_t inRank = inBatch.size();
    const std::size_t outRank = outBatch.size();
    const std::vector<int64_t> inStride = Strides(inBatch);
    int64_t offset = 0;
    for (std::size_t j = 0; j < outRank; ++j) {
        if (j + inRank < outRank) {
            continue; // a missing leading batch axis broadcasts
        }
        const std::size_t k = j - (outRank - inRank);
        if (inBatch[k] == 1) {
            continue; // a size-1 batch axis broadcasts
        }
        offset += coords[j] * inStride[k];
    }
    return offset;
}

} // namespace

void MatMul(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "MatMul (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two inputs and one output");
    }
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView& out = outputs[0];
    if (a.dtype != DataType::Float32 || b.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (a.dims.empty() || b.dims.empty()) {
        throw InferenceException(context + ": operands must have at least one dimension");
    }

    // Promote 1-D operands: A [K] -> [1,K], B [K] -> [K,1].
    std::vector<int64_t> aDims = a.dims;
    std::vector<int64_t> bDims = b.dims;
    const bool aPromoted = aDims.size() == 1;
    const bool bPromoted = bDims.size() == 1;
    if (aPromoted) {
        aDims.insert(aDims.begin(), 1);
    }
    if (bPromoted) {
        bDims.push_back(1);
    }
    (void)ElementCount(aDims, context);
    (void)ElementCount(bDims, context);

    const std::size_t aRank = aDims.size();
    const std::size_t bRank = bDims.size();
    const int64_t m = aDims[aRank - 2];
    const int64_t k = aDims[aRank - 1];
    const int64_t bK = bDims[bRank - 2];
    const int64_t n = bDims[bRank - 1];
    if (k != bK) {
        throw InferenceException(context + ": contraction dimension mismatch");
    }

    const std::vector<int64_t> batchA(aDims.begin(), aDims.end() - 2);
    const std::vector<int64_t> batchB(bDims.begin(), bDims.end() - 2);
    const std::vector<int64_t> batchOut = BroadcastDims(batchA, batchB, context);

    std::vector<int64_t> fullDims = batchOut;
    fullDims.push_back(m);
    fullDims.push_back(n);
    std::vector<int64_t> expected = fullDims;
    if (aPromoted) {
        expected.erase(expected.begin() + static_cast<std::ptrdiff_t>(batchOut.size()));
    }
    if (bPromoted) {
        const std::size_t nIndex = aPromoted ? batchOut.size() : batchOut.size() + 1;
        expected.erase(expected.begin() + static_cast<std::ptrdiff_t>(nIndex));
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the operands");
    }
    const int64_t fullCount = ElementCount(fullDims, context);

    if (a.data == nullptr || b.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (fullCount == 0) {
        return;
    }

    const float* ap = static_cast<const float*>(a.data);
    const float* bp = static_cast<const float*>(b.data);
    float* y = static_cast<float*>(out.data);
    const int64_t matSizeA = m * k;
    const int64_t matSizeB = k * n;
    const int64_t matSizeY = m * n;
    const std::vector<int64_t> batchStrides = Strides(batchOut);
    int64_t batchCount = 1;
    for (int64_t d : batchOut) {
        batchCount *= d;
    }

    cublasHandle_t handle = nullptr;
    CheckCublas(cublasCreate(&handle), "cublasCreate", context);
    // Pedantic math forbids TF32/tensor cores; DEFAULT_MATH still permits
    // TF32 on Ampere+, which the strict-float32 oracle cannot tolerate.
    CheckCublas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode", context);

    const float alpha = 1.0F;
    const float beta = 0.0F;
    for (int64_t t = 0; t < batchCount; ++t) {
        int64_t remaining = t;
        std::vector<int64_t> coords(batchOut.size(), 0);
        for (std::size_t j = 0; j < batchOut.size(); ++j) {
            coords[j] = remaining / batchStrides[j];
            remaining %= batchStrides[j];
        }
        const float* aBatch = ap + BroadcastOffset(batchA, batchOut, coords) * matSizeA;
        const float* bBatch = bp + BroadcastOffset(batchB, batchOut, coords) * matSizeB;
        float* yBatch = y + t * matSizeY;
        // Row-major swap trick, shared with Gemm: C^T = B^T * A^T.
        CheckCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m),
                                static_cast<int>(k), &alpha, bBatch, static_cast<int>(bDims[bRank - 1]), aBatch,
                                static_cast<int>(aDims[aRank - 1]), &beta, yBatch, static_cast<int>(n)),
                    "cublasSgemm", context);
    }
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
    CheckCublas(cublasDestroy(handle), "cublasDestroy", context);
}

} // namespace cuda_kernels
} // namespace nert
