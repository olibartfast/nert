// Hipified from src/cuda/arg_max.cu via hipify-perl, hand-reviewed.
// ArgMax on HIP: index of the maximum along `axis`, one thread per
// output element.
//
// Mirrors the CPU oracle: float32/int64 input, int64 output, axis default
// 0 with negative wrap, keepdims 0/1, select_last_index 0/1 tie policy.
// Each thread replays the oracle's ordered scan, so ties match exactly.

#include "nert/device.hpp"

#include <cstdint>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

namespace nert {
namespace hip_kernels {
namespace {

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

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback, const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute '" + name + "' must be an integer");
            }
            return *value;
        }
    }
    return fallback;
}

template <typename T>
__global__ void ArgMaxKernel(const T* x, int64_t* y, int64_t dim, int64_t outer, int64_t inner, bool last) {
    for (int64_t o = blockIdx.x * blockDim.x + threadIdx.x; o < outer; o += blockDim.x * gridDim.x) {
        for (int64_t n = 0; n < inner; ++n) {
            int64_t best = 0;
            T bestVal = x[(o * dim) * inner + n];
            for (int64_t i = 1; i < dim; ++i) {
                const T v = x[(o * dim + i) * inner + n];
                if (v > bestVal || (last && v == bestVal)) {
                    bestVal = v;
                    best = i;
                }
            }
            y[o * inner + n] = best;
        }
    }
}

template <typename T>
void LaunchArgMax(const T* x, int64_t* y, int64_t dim, int64_t outer, int64_t inner, bool last,
                  const std::string& context) {
    if (outer == 0) {
        return;
    }
    const int block = 256;
    const int grid = static_cast<int>((outer + block - 1) / block);
    ArgMaxKernel<<<grid, block>>>(x, y, dim, outer, inner, last);
    CheckHip(hipGetLastError(), "ArgMaxKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace

void ArgMax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ArgMax (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": input must be float32 or int64");
    }
    if (out.dtype != DataType::Int64) {
        throw InferenceException(context + ": output must be int64");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (rank == 0) {
        throw InferenceException(context + ": input must have rank >= 1");
    }
    int64_t axis = IntAttribute(node, "axis", 0, context);
    if (axis < -rank || axis >= rank) {
        throw InferenceException(context + ": axis out of range");
    }
    if (axis < 0) {
        axis += rank;
    }
    const int64_t keepdims = IntAttribute(node, "keepdims", 1, context);
    if (keepdims != 0 && keepdims != 1) {
        throw InferenceException(context + ": keepdims must be 0 or 1 here");
    }
    const int64_t last = IntAttribute(node, "select_last_index", 0, context);
    if (last != 0 && last != 1) {
        throw InferenceException(context + ": select_last_index must be 0 or 1 here");
    }
    const int64_t dim = in.dims[static_cast<std::size_t>(axis)];
    std::vector<int64_t> expected = in.dims;
    if (keepdims != 0) {
        expected[static_cast<std::size_t>(axis)] = 1;
    } else {
        expected.erase(expected.begin() + axis);
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the reduced axis");
    }
    (void)ElementCount(in.dims, context);
    const int64_t outCount = ElementCount(out.dims, context);
    if (outCount > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= in.dims[static_cast<std::size_t>(i)];
    }
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= in.dims[static_cast<std::size_t>(i)];
    }
    int64_t* y = static_cast<int64_t*>(out.data);
    const bool lastIndex = last != 0;
    if (in.dtype == DataType::Float32) {
        LaunchArgMax(static_cast<const float*>(in.data), y, dim, outer, inner, lastIndex, context);
    } else {
        LaunchArgMax(static_cast<const int64_t*>(in.data), y, dim, outer, inner, lastIndex, context);
    }
}

} // namespace hip_kernels
} // namespace nert
