// Hipified from src/cuda/less_or_equal.cu via hipify-perl, hand-reviewed.
// LessOrEqual on HIP: broadcast comparison, one thread per output element.
//
// Mirrors the CPU oracle: f32/i64 inputs sharing one dtype, bool output,
// NumPy broadcast of the inputs. Same stride template and rank cap (16) as
// Add.

#include "nert/device.hpp"

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

namespace nert {
namespace hip_kernels {
namespace {

constexpr std::size_t kMaxRank = 16;

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
__global__ void LessOrEqualKernel(const T* x, const T* z, bool* y, int64_t outCount, StrideSet outStrides,
                                  StrideSet aStrides, StrideSet bStrides, int rank) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outCount; i += blockDim.x * gridDim.x) {
        int64_t remaining = i;
        int64_t aOffset = 0;
        int64_t bOffset = 0;
        for (int j = 0; j < rank; ++j) {
            const int64_t coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            aOffset += coord * aStrides.s[j];
            bOffset += coord * bStrides.s[j];
        }
        y[i] = x[aOffset] <= z[bOffset];
    }
}

template <typename T>
void LaunchLessOrEqual(const T* x, const T* z, bool* y, int64_t outCount, const std::vector<int64_t>& outStrides,
                       const std::vector<int64_t>& aStrides, const std::vector<int64_t>& bStrides, std::size_t rank,
                       const std::string& context) {
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    LessOrEqualKernel<<<grid, block>>>(x, z, y, outCount, Pack(outStrides, context), Pack(aStrides, context),
                                       Pack(bStrides, context), static_cast<int>(rank));
    CheckHip(hipGetLastError(), "LessOrEqualKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace

void LessOrEqual(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "LessOrEqual (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two inputs and one output");
    }
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView& out = outputs[0];
    if (a.dtype != b.dtype || (a.dtype != DataType::Float32 && a.dtype != DataType::Int64)) {
        throw InferenceException(context + ": inputs must share one float32 or int64 dtype");
    }
    if (out.dtype != DataType::Bool) {
        throw InferenceException(context + ": output must be bool");
    }

    const int64_t outCount = ElementCount(out.dims, context);
    (void)ElementCount(a.dims, context);
    (void)ElementCount(b.dims, context);
    if (BroadcastDims(a.dims, b.dims, context) != out.dims) {
        throw InferenceException(context + ": output shape does not match the broadcast of the inputs");
    }
    if (a.data == nullptr || b.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (outCount == 0) {
        return;
    }
    if (out.dims.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the pilot broadcast limit of 16");
    }

    const std::vector<int64_t> outStrides = Strides(out.dims);
    const std::vector<int64_t> aStrides = AlignedStrides(a.dims, out.dims, context);
    const std::vector<int64_t> bStrides = AlignedStrides(b.dims, out.dims, context);
    if (a.dtype == DataType::Float32) {
        LaunchLessOrEqual(static_cast<const float*>(a.data), static_cast<const float*>(b.data),
                          static_cast<bool*>(out.data), outCount, outStrides, aStrides, bStrides, out.dims.size(),
                          context);
    } else {
        LaunchLessOrEqual(static_cast<const int64_t*>(a.data), static_cast<const int64_t*>(b.data),
                          static_cast<bool*>(out.data), outCount, outStrides, aStrides, bStrides, out.dims.size(),
                          context);
    }
}

} // namespace hip_kernels
} // namespace nert
