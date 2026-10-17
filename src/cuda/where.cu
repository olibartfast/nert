// Where on CUDA: broadcast ternary select, one thread per output element.
//
// Mirrors the CPU oracle: bool condition, f32/i64 values sharing one dtype
// with the output, three-way NumPy broadcast. Same stride template and rank
// cap (16) as Add.

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
__global__ void WhereKernel(const bool* c, const T* xv, const T* zv, T* y, int64_t outCount, StrideSet outStrides,
                            StrideSet condStrides, StrideSet xStrides, StrideSet zStrides, int rank) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outCount; i += blockDim.x * gridDim.x) {
        int64_t remaining = i;
        int64_t cOffset = 0;
        int64_t xOffset = 0;
        int64_t zOffset = 0;
        for (int j = 0; j < rank; ++j) {
            const int64_t coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            cOffset += coord * condStrides.s[j];
            xOffset += coord * xStrides.s[j];
            zOffset += coord * zStrides.s[j];
        }
        y[i] = c[cOffset] ? xv[xOffset] : zv[zOffset];
    }
}

template <typename T>
void LaunchWhere(const bool* c, const T* xv, const T* zv, T* y, int64_t outCount,
                 const std::vector<int64_t>& outStrides, const std::vector<int64_t>& condStrides,
                 const std::vector<int64_t>& xStrides, const std::vector<int64_t>& zStrides, std::size_t rank,
                 const std::string& context) {
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    WhereKernel<<<grid, block>>>(c, xv, zv, y, outCount, Pack(outStrides, context), Pack(condStrides, context),
                                 Pack(xStrides, context), Pack(zStrides, context), static_cast<int>(rank));
    CheckCuda(cudaGetLastError(), "WhereKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace

void Where(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Where (node '" + node.name + "')";
    if (inputs.size() != 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected three inputs and one output");
    }
    const TensorView& cond = inputs[0];
    const TensorView& x = inputs[1];
    const TensorView& z = inputs[2];
    const TensorView& out = outputs[0];
    if (cond.dtype != DataType::Bool) {
        throw InferenceException(context + ": condition must be bool");
    }
    if (x.dtype != z.dtype || (x.dtype != DataType::Float32 && x.dtype != DataType::Int64) || out.dtype != x.dtype) {
        throw InferenceException(context + ": value tensors must share one float32 or int64 dtype");
    }

    const int64_t outCount = ElementCount(out.dims, context);
    (void)ElementCount(cond.dims, context);
    (void)ElementCount(x.dims, context);
    (void)ElementCount(z.dims, context);
    if (BroadcastDims(BroadcastDims(cond.dims, x.dims, context), z.dims, context) != out.dims) {
        throw InferenceException(context + ": output shape does not match the broadcast of the inputs");
    }
    if (cond.data == nullptr || x.data == nullptr || z.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (outCount == 0) {
        return;
    }
    if (out.dims.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the pilot broadcast limit of 16");
    }

    const std::vector<int64_t> outStrides = Strides(out.dims);
    const std::vector<int64_t> condStrides = AlignedStrides(cond.dims, out.dims, context);
    const std::vector<int64_t> xStrides = AlignedStrides(x.dims, out.dims, context);
    const std::vector<int64_t> zStrides = AlignedStrides(z.dims, out.dims, context);
    if (x.dtype == DataType::Float32) {
        LaunchWhere(static_cast<const bool*>(cond.data), static_cast<const float*>(x.data),
                    static_cast<const float*>(z.data), static_cast<float*>(out.data), outCount, outStrides, condStrides,
                    xStrides, zStrides, out.dims.size(), context);
    } else {
        LaunchWhere(static_cast<const bool*>(cond.data), static_cast<const int64_t*>(x.data),
                    static_cast<const int64_t*>(z.data), static_cast<int64_t*>(out.data), outCount, outStrides,
                    condStrides, xStrides, zStrides, out.dims.size(), context);
    }
}

} // namespace cuda_kernels
} // namespace nert
