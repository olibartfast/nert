// Min on CUDA: elementwise minimum over one or more broadcast inputs.
//
// Mirrors the CPU oracle: one shared float32/int64 dtype, the full N-way
// NumPy broadcast validated upfront against the output shape. Execution
// folds pairwise in input order (broadcast-copy input0, then fold each
// further input into the output buffer), which is bit-identical to the
// oracle's ordered fold, including NaN order. Rank cap 16 (pilot limit).

#include "nert/device.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace nert {
namespace cuda_kernels {
namespace {

constexpr std::size_t kMaxRank = 16;

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

std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

std::vector<int64_t> BroadcastDims(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                                   const std::string& context) {
    const std::size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank, 1);
    for (std::size_t i = 0; i < rank; ++i) {
        const int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
        const int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
        if (da != db && da != 1 && db != 1) {
            throw InferenceException(context + ": shapes are not broadcast-compatible");
        }
        out[i] = std::max(da, db);
    }
    return out;
}

// For each output axis, the input stride that maps it (0 where the input axis
// is broadcast). Throws when the input shape cannot broadcast to the output.
std::vector<int64_t> AlignedStrides(const std::vector<int64_t>& inDims, const std::vector<int64_t>& outDims,
                                    const std::string& context) {
    const std::size_t outRank = outDims.size();
    const std::size_t inRank = inDims.size();
    const std::vector<int64_t> inStrides = Strides(inDims);
    std::vector<int64_t> aligned(outRank, 0);
    for (std::size_t j = 0; j < outRank; ++j) {
        if (j + inRank < outRank) {
            continue; // a missing leading axis broadcasts as a size-1 axis
        }
        const std::size_t k = j - (outRank - inRank);
        if (inDims[k] == outDims[j]) {
            aligned[j] = inStrides[k];
        } else if (inDims[k] == 1) {
            aligned[j] = 0; // broadcast
        } else {
            throw InferenceException(context + ": shapes are not broadcast-compatible");
        }
    }
    return aligned;
}

struct StrideSet {
    int64_t s[kMaxRank];
};

StrideSet Pack(const std::vector<int64_t>& strides, const std::string& context) {
    if (strides.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the pilot broadcast limit of 16");
    }
    StrideSet packed{};
    for (std::size_t j = 0; j < strides.size(); ++j) {
        packed.s[j] = strides[j];
    }
    return packed;
}

template <typename T>
__global__ void BroadcastCopyKernel(const T* x, T* y, int64_t outCount, StrideSet outStrides, StrideSet inStrides,
                                    int rank) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outCount; i += blockDim.x * gridDim.x) {
        int64_t remaining = i;
        int64_t offset = 0;
        for (int j = 0; j < rank; ++j) {
            const int64_t coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            offset += coord * inStrides.s[j];
        }
        y[i] = x[offset];
    }
}

template <typename T>
__global__ void MinFoldKernel(const T* acc, const T* x, T* y, int64_t outCount, StrideSet outStrides,
                              StrideSet inStrides, int rank) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outCount; i += blockDim.x * gridDim.x) {
        int64_t remaining = i;
        int64_t offset = 0;
        for (int j = 0; j < rank; ++j) {
            const int64_t coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            offset += coord * inStrides.s[j];
        }
        const T current = acc[i];
        const T v = x[offset];
        y[i] = current > v ? v : current;
    }
}

template <typename T>
void RunMin(const T* first, T* out, int64_t outCount, const std::vector<TensorView>& rest,
            const std::vector<int64_t>& outStrides, const std::vector<std::vector<int64_t>>& restStrides,
            std::size_t rank, const std::vector<int64_t>& firstStrides, const std::string& context) {
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    BroadcastCopyKernel<<<grid, block>>>(first, out, outCount, Pack(outStrides, context), Pack(firstStrides, context),
                                         static_cast<int>(rank));
    CheckCuda(cudaGetLastError(), "BroadcastCopyKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
    for (std::size_t k = 0; k < rest.size(); ++k) {
        MinFoldKernel<<<grid, block>>>(out, static_cast<const T*>(rest[k].data), out, outCount,
                                       Pack(outStrides, context), Pack(restStrides[k], context),
                                       static_cast<int>(rank));
        CheckCuda(cudaGetLastError(), "MinFoldKernel launch", context);
        CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
    }
}

} // namespace

void Min(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Min (node '" + node.name + "')";
    if (inputs.empty() || outputs.size() != 1) {
        throw InferenceException(context + ": expected at least one input and one output");
    }
    const DataType dtype = inputs[0].dtype;
    if (dtype != DataType::Float32 && dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must share one float32 or int64 dtype");
    }
    for (const TensorView& in : inputs) {
        if (in.dtype != dtype) {
            throw InferenceException(context + ": tensors must share one float32 or int64 dtype");
        }
    }
    const TensorView& out = outputs[0];
    if (out.dtype != dtype) {
        throw InferenceException(context + ": tensors must share one float32 or int64 dtype");
    }

    std::vector<int64_t> acc = inputs[0].dims;
    (void)ElementCount(inputs[0].dims, context);
    for (std::size_t i = 1; i < inputs.size(); ++i) {
        (void)ElementCount(inputs[i].dims, context);
        acc = BroadcastDims(acc, inputs[i].dims, context);
    }
    if (acc != out.dims) {
        throw InferenceException(context + ": output shape does not match the broadcast of the inputs");
    }
    const int64_t outCount = ElementCount(out.dims, context);
    for (const TensorView& in : inputs) {
        if (in.data == nullptr) {
            throw InferenceException(context + ": null buffer");
        }
    }
    if (out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (outCount == 0) {
        return;
    }
    if (out.dims.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the pilot broadcast limit of 16");
    }

    const std::vector<int64_t> outStrides = Strides(out.dims);
    const std::vector<int64_t> firstStrides = AlignedStrides(inputs[0].dims, out.dims, context);
    std::vector<TensorView> rest(inputs.begin() + 1, inputs.end());
    std::vector<std::vector<int64_t>> restStrides;
    for (const TensorView& in : rest) {
        restStrides.push_back(AlignedStrides(in.dims, out.dims, context));
    }
    if (dtype == DataType::Float32) {
        RunMin(static_cast<const float*>(inputs[0].data), static_cast<float*>(out.data), outCount, rest, outStrides,
               restStrides, out.dims.size(), firstStrides, context);
    } else {
        RunMin(static_cast<const int64_t*>(inputs[0].data), static_cast<int64_t*>(out.data), outCount, rest, outStrides,
               restStrides, out.dims.size(), firstStrides, context);
    }
}

} // namespace cuda_kernels
} // namespace nert
