// Hipified from src/cuda/transpose.cu via hipify-perl, hand-reviewed.
// Transpose on HIP: permuted copy, one thread per output element.
//
// Mirrors the CPU oracle: f32/i64 sharing one dtype, perm attr defaulting
// to the reversed rank, permutation + output-shape checks. Ranks above 16
// are rejected (pilot limit, shared with Add).

#include "nert/device.hpp"

#include <algorithm>
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

struct IndexSet {
    int64_t s[kMaxRank];
};

IndexSet Pack(const std::vector<int64_t>& values, const std::string& context) {
    if (values.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the transpose limit of 16");
    }
    IndexSet packed{};
    for (std::size_t j = 0; j < values.size(); ++j) {
        packed.s[j] = values[j];
    }
    return packed;
}

template <typename T>
__global__ void TransposeKernel(const T* x, T* y, int64_t outCount, IndexSet perm, IndexSet inStrides,
                                IndexSet outStrides, int rank) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outCount; i += blockDim.x * gridDim.x) {
        int64_t remaining = i;
        int64_t offset = 0;
        for (int j = 0; j < rank; ++j) {
            const int64_t coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            offset += coord * inStrides.s[perm.s[j]];
        }
        y[i] = x[offset];
    }
}

template <typename T>
void LaunchTranspose(const T* x, T* y, const std::vector<int64_t>& perm, const std::vector<int64_t>& inStrides,
                     const std::vector<int64_t>& outStrides, int64_t outCount, int64_t rank,
                     const std::string& context) {
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    TransposeKernel<<<grid, block>>>(x, y, outCount, Pack(perm, context), Pack(inStrides, context),
                                     Pack(outStrides, context), static_cast<int>(rank));
    CheckHip(hipGetLastError(), "TransposeKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace

void Transpose(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Transpose (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool) {
        throw InferenceException(context + ": tensors must be float32, int64, or bool");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (static_cast<std::size_t>(rank) > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the transpose limit of 16");
    }

    std::vector<int64_t> perm;
    bool have_perm = false;
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "perm") {
            const std::vector<int64_t>* value = std::get_if<std::vector<int64_t>>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'perm' must be an int list");
            }
            perm = *value;
            have_perm = true;
        }
    }
    if (!have_perm) {
        perm.resize(static_cast<std::size_t>(rank));
        for (int64_t i = 0; i < rank; ++i) {
            perm[static_cast<std::size_t>(i)] = rank - 1 - i;
        }
    }
    if (static_cast<int64_t>(perm.size()) != rank) {
        throw InferenceException(context + ": Transpose perm rank must match the input rank");
    }
    std::vector<int64_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (int64_t i = 0; i < rank; ++i) {
        if (sorted[static_cast<std::size_t>(i)] != i) {
            throw InferenceException(context + ": Transpose perm must be a permutation");
        }
    }
    std::vector<int64_t> expected(static_cast<std::size_t>(rank), 0);
    for (int64_t i = 0; i < rank; ++i) {
        expected[static_cast<std::size_t>(i)] = in.dims[static_cast<std::size_t>(perm[i])];
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the transposed input");
    }

    const int64_t count = ElementCount(out.dims, context);
    (void)ElementCount(in.dims, context);
    if (count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    if (count == 0) {
        return;
    }

    const std::vector<int64_t> in_strides = Strides(in.dims);
    const std::vector<int64_t> out_strides = Strides(out.dims);
    if (in.dtype == DataType::Float32) {
        LaunchTranspose(static_cast<const float*>(in.data), static_cast<float*>(out.data), perm, in_strides,
                        out_strides, count, rank, context);
    } else if (in.dtype == DataType::Int64) {
        LaunchTranspose(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), perm, in_strides,
                        out_strides, count, rank, context);
    } else {
        LaunchTranspose(static_cast<const bool*>(in.data), static_cast<bool*>(out.data), perm, in_strides, out_strides,
                        count, rank, context);
    }
}

} // namespace hip_kernels
} // namespace nert
