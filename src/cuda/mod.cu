// Mod on CUDA: broadcast integer remainder, one thread per output element.
//
// Mirrors the CPU oracle: int64-only, fmod attr restricted to 0, NumPy
// broadcast of the inputs. A zero divisor is flagged on device and reported
// on the host (Div precedent). Same stride template and rank cap (16) as
// Add.

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

__global__ void ModKernel(const int64_t* x, const int64_t* z, int64_t* y, int64_t outCount, StrideSet outStrides,
                          StrideSet aStrides, StrideSet bStrides, int rank, int* zeroFlag) {
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
        if (z[bOffset] == 0) {
            atomicExch(zeroFlag, 1);
            y[i] = 0;
        } else {
            y[i] = x[aOffset] % z[bOffset];
        }
    }
}

} // namespace

void Mod(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Mod (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two inputs and one output");
    }
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "fmod") {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'fmod' must be an integer");
            }
            if (*value != 0) {
                throw InferenceException(context + ": only fmod=0 (integer remainder) is supported");
            }
        }
    }
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView& out = outputs[0];
    if (a.dtype != DataType::Int64 || b.dtype != DataType::Int64 || out.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be int64");
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
    int* zeroFlag = nullptr;
    CheckCuda(cudaMalloc(&zeroFlag, sizeof(int)), "cudaMalloc", context);
    CheckCuda(cudaMemset(zeroFlag, 0, sizeof(int)), "cudaMemset", context);
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    ModKernel<<<grid, block>>>(static_cast<const int64_t*>(a.data), static_cast<const int64_t*>(b.data),
                               static_cast<int64_t*>(out.data), outCount, Pack(outStrides, context),
                               Pack(aStrides, context), Pack(bStrides, context), static_cast<int>(out.dims.size()),
                               zeroFlag);
    CheckCuda(cudaGetLastError(), "ModKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
    int flagged = 0;
    CheckCuda(cudaMemcpy(&flagged, zeroFlag, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy", context);
    CheckCuda(cudaFree(zeroFlag), "cudaFree", context);
    if (flagged != 0) {
        throw InferenceException(context + ": division by zero");
    }
}

} // namespace cuda_kernels
} // namespace nert
