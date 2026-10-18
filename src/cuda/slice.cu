// Slice on CUDA: strided sub-tensor copy, one thread per output element.
//
// Mirrors the CPU oracle: f32/i64 sharing one dtype, starts/ends (+ optional
// axes/steps) operands downloaded from device, negative wrap, clamping,
// positive and negative steps, empty selections carrying only the shape
// contract. Ranks above 16 are rejected (pilot limit, shared with Add).

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

std::vector<int64_t> ReadInt64s(const TensorView& view, const std::string& what, const std::string& context) {
    if (view.dtype != DataType::Int64) {
        throw InferenceException(context + ": Slice " + what + " must be int64");
    }
    const int64_t count = ElementCount(view.dims, context);
    if (count > 0 && view.data == nullptr) {
        throw InferenceException(context + ": null " + what + " buffer");
    }
    std::vector<int64_t> values(static_cast<std::size_t>(count), 0);
    if (count > 0) {
        CheckCuda(cudaMemcpy(values.data(), view.data, static_cast<std::size_t>(count) * sizeof(int64_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy", context);
    }
    return values;
}

struct StrideSet {
    int64_t s[kMaxRank];
};

StrideSet Pack(const std::vector<int64_t>& strides, const std::string& context) {
    if (strides.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the slice limit of 16");
    }
    StrideSet packed{};
    for (std::size_t j = 0; j < strides.size(); ++j) {
        packed.s[j] = strides[j];
    }
    return packed;
}

template <typename T>
__global__ void SliceKernel(const T* x, T* y, int64_t outCount, StrideSet begins, StrideSet steps, StrideSet inStrides,
                            StrideSet outStrides, int rank) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outCount; i += blockDim.x * gridDim.x) {
        int64_t remaining = i;
        int64_t offset = 0;
        for (int j = 0; j < rank; ++j) {
            const int64_t coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            offset += (begins.s[j] + coord * steps.s[j]) * inStrides.s[j];
        }
        y[i] = x[offset];
    }
}

template <typename T>
void LaunchSlice(const T* x, T* y, const std::vector<int64_t>& begins, const std::vector<int64_t>& steps,
                 const std::vector<int64_t>& inStrides, const std::vector<int64_t>& outStrides, int64_t outCount,
                 int64_t rank, const std::string& context) {
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    SliceKernel<<<grid, block>>>(x, y, outCount, Pack(begins, context), Pack(steps, context), Pack(inStrides, context),
                                 Pack(outStrides, context), static_cast<int>(rank));
    CheckCuda(cudaGetLastError(), "SliceKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace

void Slice(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Slice (node '" + node.name + "')";
    if (inputs.size() < 3 || inputs.size() > 5 || outputs.size() != 1) {
        throw InferenceException(context + ": expected data, starts, ends, and optional axes/steps inputs");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool) {
        throw InferenceException(context + ": tensors must be float32, int64, or bool");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    const std::vector<int64_t> starts = ReadInt64s(inputs[1], "starts", context);
    const std::vector<int64_t> ends = ReadInt64s(inputs[2], "ends", context);
    if (starts.size() != ends.size()) {
        throw InferenceException(context + ": Slice starts and ends must have the same length");
    }
    const std::size_t count = starts.size();

    std::vector<int64_t> axes(count, 0);
    for (std::size_t i = 0; i < count; ++i) {
        axes[i] = static_cast<int64_t>(i);
    }
    if (inputs.size() > 3 && inputs[3].dtype != DataType::Unknown) {
        axes = ReadInt64s(inputs[3], "axes", context);
        if (axes.size() != count) {
            throw InferenceException(context + ": Slice axes must match starts in length");
        }
    }
    std::vector<int64_t> steps(count, 1);
    if (inputs.size() > 4 && inputs[4].dtype != DataType::Unknown) {
        steps = ReadInt64s(inputs[4], "steps", context);
        if (steps.size() != count) {
            throw InferenceException(context + ": Slice steps must match starts in length");
        }
    }

    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (static_cast<std::size_t>(rank) > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the slice limit of 16");
    }
    std::vector<int64_t> begins(static_cast<std::size_t>(rank), 0);
    std::vector<int64_t> strides(static_cast<std::size_t>(rank), 1);
    std::vector<int64_t> expected = in.dims;
    for (std::size_t i = 0; i < count; ++i) {
        const int64_t step = steps[i];
        if (step == 0) {
            throw InferenceException(context + ": Slice steps must be non-zero");
        }
        int64_t axis = axes[i];
        if (axis < -rank || axis >= rank) {
            throw InferenceException(context + ": Slice axis is out of range");
        }
        if (axis < 0) {
            axis += rank;
        }
        const std::size_t ua = static_cast<std::size_t>(axis);
        const int64_t dim = in.dims[ua];
        int64_t begin = starts[i];
        int64_t end = ends[i];
        if (begin < 0) {
            begin += dim;
        }
        if (end < 0) {
            end += dim;
        }
        if (step > 0) {
            begin = std::max<int64_t>(0, std::min(begin, dim));
            end = std::max<int64_t>(0, std::min(end, dim));
            expected[ua] = end > begin ? (end - begin + step - 1) / step : 0;
        } else {
            begin = std::max<int64_t>(0, std::min(begin, dim - 1));
            end = std::max<int64_t>(-1, std::min(end, dim - 1));
            expected[ua] = dim > 0 && begin > end ? (begin - end - step - 1) / -step : 0;
        }
        begins[ua] = begin;
        strides[ua] = step;
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the sliced ranges");
    }

    // Empty selections carry no elements; only the shape contract applies.
    int64_t out_count = 1;
    for (const int64_t d : out.dims) {
        if (d < 0) {
            throw InferenceException(context + ": non-positive dimension");
        }
        out_count *= d;
    }
    (void)ElementCount(in.dims, context);
    if (out_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    if (out_count == 0) {
        return;
    }

    const std::vector<int64_t> in_strides = Strides(in.dims);
    const std::vector<int64_t> out_strides = Strides(out.dims);
    if (in.dtype == DataType::Float32) {
        LaunchSlice(static_cast<const float*>(in.data), static_cast<float*>(out.data), begins, strides, in_strides,
                    out_strides, out_count, rank, context);
    } else if (in.dtype == DataType::Int64) {
        LaunchSlice(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), begins, strides, in_strides,
                    out_strides, out_count, rank, context);
    } else {
        LaunchSlice(static_cast<const bool*>(in.data), static_cast<bool*>(out.data), begins, strides, in_strides,
                    out_strides, out_count, rank, context);
    }
}

} // namespace cuda_kernels
} // namespace nert
