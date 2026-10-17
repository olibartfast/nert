// Expand on CUDA: broadcast the input to the target shape, one thread per
// output element.
//
// Mirrors the CPU oracle: f32/i64/bool sharing one dtype, int64 shape
// operand (downloaded from device), positive target dims, NumPy-style
// broadcastability with the shorter shape right-aligned. Strides resolve on
// the host into rank-indexed arrays; ranks above 16 are rejected like Add.

#include "nert/device.hpp"

#include <algorithm>
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

struct StrideSet {
    int64_t s[kMaxRank];
};

StrideSet Pack(const std::vector<int64_t>& strides, const std::string& context) {
    if (strides.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the broadcast limit of 16");
    }
    StrideSet packed{};
    for (std::size_t j = 0; j < strides.size(); ++j) {
        packed.s[j] = strides[j];
    }
    return packed;
}

template <typename T>
__global__ void ExpandKernel(const T* x, T* y, int64_t outCount, StrideSet outStrides, StrideSet aligned, int rank) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outCount; i += blockDim.x * gridDim.x) {
        int64_t remaining = i;
        int64_t offset = 0;
        for (int j = 0; j < rank; ++j) {
            const int64_t coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            offset += coord * aligned.s[j];
        }
        y[i] = x[offset];
    }
}

template <typename T>
void LaunchExpand(const T* x, T* y, int64_t outCount, const std::vector<int64_t>& outStrides,
                  const std::vector<int64_t>& aligned, std::size_t rank, const std::string& context) {
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    ExpandKernel<<<grid, block>>>(x, y, outCount, Pack(outStrides, context), Pack(aligned, context),
                                  static_cast<int>(rank));
    CheckCuda(cudaGetLastError(), "ExpandKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace

void Expand(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Expand (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected a data input, a shape input, and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& shape_view = inputs[1];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool) {
        throw InferenceException(context + ": tensors must be float32, int64, or bool");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    if (shape_view.dtype != DataType::Int64) {
        throw InferenceException(context + ": Expand requires a shape input");
    }
    const int64_t target_count = ElementCount(shape_view.dims, context);
    if (target_count > 0 && shape_view.data == nullptr) {
        throw InferenceException(context + ": null shape buffer");
    }
    std::vector<int64_t> target(static_cast<std::size_t>(target_count), 0);
    if (target_count > 0) {
        CheckCuda(cudaMemcpy(target.data(), shape_view.data, static_cast<std::size_t>(target_count) * sizeof(int64_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy", context);
    }

    const std::size_t rank = std::max(in.dims.size(), target.size());
    if (rank > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the broadcast limit of 16");
    }
    std::vector<int64_t> expected(rank, 1);
    for (std::size_t i = 0; i < rank; ++i) {
        const int64_t di = static_cast<int64_t>(in.dims.size()) - static_cast<int64_t>(rank) + static_cast<int64_t>(i);
        const int64_t si = static_cast<int64_t>(target.size()) - static_cast<int64_t>(rank) + static_cast<int64_t>(i);
        const int64_t d = di < 0 ? 1 : in.dims[static_cast<std::size_t>(di)];
        if (si < 0) {
            expected[i] = d;
            continue;
        }
        const int64_t s = target[static_cast<std::size_t>(si)];
        if (s < 1) {
            throw InferenceException(context + ": Expand target shape must be positive");
        }
        if (d == s) {
            expected[i] = s;
        } else if (d == 1 || di < 0) {
            expected[i] = s;
        } else if (s == 1) {
            expected[i] = d;
        } else {
            throw InferenceException(context + ": Expand input is not broadcastable to the target shape");
        }
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the expanded shape");
    }

    const int64_t out_count = ElementCount(out.dims, context);
    (void)ElementCount(in.dims, context);
    if (out_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    if (out_count == 0) {
        return;
    }

    const std::vector<int64_t> in_strides = Strides(in.dims);
    const std::vector<int64_t> out_strides = Strides(out.dims);
    const int64_t out_rank = static_cast<int64_t>(rank);
    const int64_t in_rank = static_cast<int64_t>(in.dims.size());
    std::vector<int64_t> aligned(static_cast<std::size_t>(out_rank), 0);
    for (int64_t j = 0; j < out_rank; ++j) {
        const int64_t k = j - (out_rank - in_rank);
        if (k < 0 || in.dims[static_cast<std::size_t>(k)] == 1) {
            aligned[static_cast<std::size_t>(j)] = 0;
        } else {
            aligned[static_cast<std::size_t>(j)] = in_strides[static_cast<std::size_t>(k)];
        }
    }
    if (in.dtype == DataType::Float32) {
        LaunchExpand(static_cast<const float*>(in.data), static_cast<float*>(out.data), out_count, out_strides, aligned,
                     rank, context);
    } else if (in.dtype == DataType::Int64) {
        LaunchExpand(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), out_count, out_strides,
                     aligned, rank, context);
    } else {
        LaunchExpand(static_cast<const bool*>(in.data), static_cast<bool*>(out.data), out_count, out_strides, aligned,
                     rank, context);
    }
}

} // namespace cuda_kernels
} // namespace nert
