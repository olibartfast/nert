// Tile on CUDA: repeat the input along every axis by the repeats input.
//
// Mirrors the CPU oracle: float32/int64/bool sharing one dtype, repeats a
// rank-1 int64 tensor with one entry per input axis, output dim i = input
// dim i times repeats[i]. One thread per output element reads the input at
// coordinate modulo the input dim. Rank cap 16 (pilot limit).

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

struct IndexSet {
    int64_t s[kMaxRank];
};

IndexSet Pack(const std::vector<int64_t>& values, const std::string& context) {
    if (values.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the tile limit of 16");
    }
    IndexSet packed{};
    for (std::size_t j = 0; j < values.size(); ++j) {
        packed.s[j] = values[j];
    }
    return packed;
}

template <typename T>
__global__ void TileKernel(const T* x, T* y, int64_t outCount, IndexSet inDims, IndexSet inStrides, IndexSet outStrides,
                           int rank) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outCount; i += blockDim.x * gridDim.x) {
        int64_t remaining = i;
        int64_t offset = 0;
        for (int j = 0; j < rank; ++j) {
            const int64_t coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            offset += (coord % inDims.s[j]) * inStrides.s[j];
        }
        y[i] = x[offset];
    }
}

template <typename T>
void LaunchTile(const T* x, T* y, int64_t outCount, const std::vector<int64_t>& inDims,
                const std::vector<int64_t>& inStrides, const std::vector<int64_t>& outStrides, std::size_t rank,
                const std::string& context) {
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    TileKernel<<<grid, block>>>(x, y, outCount, Pack(inDims, context), Pack(inStrides, context),
                                Pack(outStrides, context), static_cast<int>(rank));
    CheckCuda(cudaGetLastError(), "TileKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace

void Tile(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Tile (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected a data input, a repeats input, and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& repeats = inputs[1];
    const TensorView& out = outputs[0];
    if (in.dtype != out.dtype ||
        (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool)) {
        throw InferenceException(context + ": data must share one float32, int64, or bool dtype");
    }
    if (repeats.dtype != DataType::Int64 || repeats.dims.size() != 1 ||
        repeats.dims[0] != static_cast<int64_t>(in.dims.size())) {
        throw InferenceException(context + ": repeats must be a rank-1 int64 tensor with one entry per input axis");
    }
    (void)ElementCount(in.dims, context);
    const int64_t outCount = ElementCount(out.dims, context);
    if (in.data == nullptr || repeats.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (in.dims.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the tile limit of 16");
    }
    std::vector<int64_t> reps(static_cast<std::size_t>(repeats.dims[0]), 0);
    CheckCuda(cudaMemcpy(reps.data(), repeats.data, reps.size() * sizeof(int64_t), cudaMemcpyDeviceToHost),
              "cudaMemcpy", context);
    if (out.dims.size() != in.dims.size()) {
        throw InferenceException(context + ": output rank does not match the input rank");
    }
    for (std::size_t j = 0; j < in.dims.size(); ++j) {
        if (reps[j] <= 0 || out.dims[j] != in.dims[j] * reps[j]) {
            throw InferenceException(context + ": output shape does not match input dims times repeats");
        }
    }
    if (outCount == 0) {
        return;
    }

    const std::vector<int64_t> inStrides = Strides(in.dims);
    const std::vector<int64_t> outStrides = Strides(out.dims);
    if (in.dtype == DataType::Float32) {
        LaunchTile(static_cast<const float*>(in.data), static_cast<float*>(out.data), outCount, in.dims, inStrides,
                   outStrides, in.dims.size(), context);
    } else if (in.dtype == DataType::Int64) {
        LaunchTile(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), outCount, in.dims, inStrides,
                   outStrides, in.dims.size(), context);
    } else {
        LaunchTile(static_cast<const bool*>(in.data), static_cast<bool*>(out.data), outCount, in.dims, inStrides,
                   outStrides, in.dims.size(), context);
    }
}

} // namespace cuda_kernels
} // namespace nert
