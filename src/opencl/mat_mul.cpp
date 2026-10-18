// MatMul on OpenCL: ONNX matrix product with 1-D promotion and NumPy batch
// broadcast, one work-item per output element.
//
// Mirrors the CPU oracle's validation exactly (float32 only, rank >= 1,
// 1-D promotion, contraction match, batch broadcast, output dims). Batch
// coordinates resolve on the host into per-axis dims/strides packed in
// by-value structs with a fixed max batch rank of 8; a missing leading
// axis and a size-1 axis both broadcast with stride 0, like the CPU
// BroadcastOffset. Each work item accumulates its dot product in float
// and stores once, so every output element rounds once to float32.

#include "nert/device.hpp"
#include "runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

namespace nert {
namespace opencl_kernels {
namespace {

constexpr std::size_t kMaxBatchRank = 8;

void CheckCl(cl_int status, const char* what, const std::string& context) {
    if (status != CL_SUCCESS) {
        throw InferenceException(context + ": " + what + " failed (OpenCL status " + std::to_string(status) + ")");
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

// Per output batch axis, the input's batch stride in elements: the input's
// own row-major batch stride scaled by the matrix tile size, or 0 when the
// axis broadcasts (a missing leading axis or a size-1 axis), like the CPU
// BroadcastOffset.
std::vector<int64_t> AlignedBatchStrides(const std::vector<int64_t>& inBatch, const std::vector<int64_t>& outBatch,
                                         int64_t matSize) {
    const std::size_t outRank = outBatch.size();
    const std::size_t inRank = inBatch.size();
    const std::vector<int64_t> inStrides = Strides(inBatch);
    std::vector<int64_t> aligned(outRank, 0);
    for (std::size_t j = 0; j < outRank; ++j) {
        if (j + inRank < outRank) {
            continue; // a missing leading batch axis broadcasts
        }
        const std::size_t k = j - (outRank - inRank);
        if (inBatch[k] == 1) {
            continue; // a size-1 batch axis broadcasts
        }
        aligned[j] = inStrides[k] * matSize;
    }
    return aligned;
}

// Batch dims plus per-axis strides, passed to the kernel by value.
struct BatchInfo {
    cl_long dims[kMaxBatchRank];
    cl_long strides[kMaxBatchRank];
};

BatchInfo PackBatch(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides, const std::string& context) {
    if (dims.size() > kMaxBatchRank) {
        throw InferenceException(context + ": batch rank exceeds the OpenCL MatMul limit of 8");
    }
    BatchInfo packed{};
    for (std::size_t j = 0; j < dims.size(); ++j) {
        packed.dims[j] = static_cast<cl_long>(dims[j]);
        packed.strides[j] = static_cast<cl_long>(strides[j]);
    }
    return packed;
}

constexpr const char* kSource = R"CLC(
typedef struct { long dims[8]; long strides[8]; } BatchInfo;
__kernel void mat_mul_f(__global const float* a, __global const float* b, __global float* y, long count, long m,
                        long k, long n, int batchRank, BatchInfo outBatch, BatchInfo aBatch, BatchInfo bBatch) {
    long i = get_global_id(0);
    if (i < count) {
        long ni = i % n;
        long mi = (i / n) % m;
        long remaining = i / (m * n);
        long aOffset = 0;
        long bOffset = 0;
        for (int j = batchRank - 1; j >= 0; --j) {
            long coord = remaining % outBatch.dims[j];
            remaining /= outBatch.dims[j];
            aOffset += coord * aBatch.strides[j];
            bOffset += coord * bBatch.strides[j];
        }
        __global const float* aMat = a + aOffset + mi * k;
        __global const float* bMat = b + bOffset + ni;
        float acc = 0.0f;
        for (long kk = 0; kk < k; ++kk) {
            acc += aMat[kk] * bMat[kk * n];
        }
        y[i] = acc;
    }
}
)CLC";

void Launch(cl_kernel kernel, cl_mem a, cl_mem b, cl_mem y, int64_t outCount, int64_t m, int64_t k, int64_t n,
            int batchRank, const BatchInfo& outBatch, const BatchInfo& aBatch, const BatchInfo& bBatch,
            const std::string& context) {
    cl_long count = static_cast<cl_long>(outCount);
    cl_long mArg = static_cast<cl_long>(m);
    cl_long kArg = static_cast<cl_long>(k);
    cl_long nArg = static_cast<cl_long>(n);
    cl_int rankArg = static_cast<cl_int>(batchRank);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(a), &a), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(b), &b), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(y), &y), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(count), &count), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(mArg), &mArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 5, sizeof(kArg), &kArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 6, sizeof(nArg), &nArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 7, sizeof(rankArg), &rankArg), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 8, sizeof(outBatch), &outBatch), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 9, sizeof(aBatch), &aBatch), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 10, sizeof(bBatch), &bBatch), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(outCount);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
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

    const int64_t matSizeA = m * k;
    const int64_t matSizeB = k * n;
    const BatchInfo outInfo = PackBatch(batchOut, Strides(batchOut), context);
    const BatchInfo aInfo = PackBatch(batchA, AlignedBatchStrides(batchA, batchOut, matSizeA), context);
    const BatchInfo bInfo = PackBatch(batchB, AlignedBatchStrides(batchB, batchOut, matSizeB), context);

    cl_mem da = reinterpret_cast<cl_mem>(const_cast<void*>(a.data));
    cl_mem db = reinterpret_cast<cl_mem>(const_cast<void*>(b.data));
    cl_mem dy = reinterpret_cast<cl_mem>(out.data);
    Launch(opencl_runtime::KernelFor("MatMul", kSource, "mat_mul_f"), da, db, dy, fullCount, m, k, n,
           static_cast<int>(batchOut.size()), outInfo, aInfo, bInfo, context);
}

} // namespace opencl_kernels
} // namespace nert
