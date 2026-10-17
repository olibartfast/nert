// CumSum on CUDA: cumulative sum along the axis input.
//
// Mirrors the CPU oracle: float32 (double running sum) or int64 input,
// shared output dtype and shape, scalar int64 axis input downloaded to the
// host, exclusive/reverse 0/1 attrs. One thread per (outer, inner) lane
// replays the oracle's ordered accumulation exactly.

#include "nert/device.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace nert {
namespace cuda_kernels {
namespace {

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

template <typename T, typename Acc>
__global__ void CumSumKernel(const T* x, T* y, int64_t axisDim, int64_t outer, int64_t inner, bool exclusive,
                             bool reverse) {
    for (int64_t o = blockIdx.x * blockDim.x + threadIdx.x; o < outer; o += blockDim.x * gridDim.x) {
        for (int64_t n = 0; n < inner; ++n) {
            Acc run{};
            if (!reverse) {
                for (int64_t i = 0; i < axisDim; ++i) {
                    const int64_t idx = (o * axisDim + i) * inner + n;
                    if (exclusive) {
                        y[idx] = static_cast<T>(run);
                        run += static_cast<Acc>(x[idx]);
                    } else {
                        run += static_cast<Acc>(x[idx]);
                        y[idx] = static_cast<T>(run);
                    }
                }
            } else {
                for (int64_t i = axisDim - 1; i >= 0; --i) {
                    const int64_t idx = (o * axisDim + i) * inner + n;
                    if (exclusive) {
                        y[idx] = static_cast<T>(run);
                        run += static_cast<Acc>(x[idx]);
                    } else {
                        run += static_cast<Acc>(x[idx]);
                        y[idx] = static_cast<T>(run);
                    }
                }
            }
        }
    }
}

template <typename T, typename Acc>
void LaunchCumSum(const T* x, T* y, int64_t axisDim, int64_t outer, int64_t inner, bool exclusive, bool reverse,
                  const std::string& context) {
    if (outer == 0) {
        return;
    }
    const int block = 256;
    const int grid = static_cast<int>((outer + block - 1) / block);
    CumSumKernel<T, Acc><<<grid, block>>>(x, y, axisDim, outer, inner, exclusive, reverse);
    CheckCuda(cudaGetLastError(), "CumSumKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace

void CumSum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "CumSum (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected a data input, an axis input, and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& axisView = inputs[1];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": input must be float32 or int64");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output must share the input dtype");
    }
    if (axisView.dtype != DataType::Int64) {
        throw InferenceException(context + ": axis input must be int64");
    }
    if (ElementCount(axisView.dims, context) != 1) {
        throw InferenceException(context + ": axis input must be a scalar");
    }
    if (axisView.data == nullptr) {
        throw InferenceException(context + ": null axis buffer");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (rank == 0) {
        throw InferenceException(context + ": input must have rank >= 1");
    }
    int64_t axis = 0;
    CheckCuda(cudaMemcpy(&axis, axisView.data, sizeof(int64_t), cudaMemcpyDeviceToHost), "cudaMemcpy", context);
    if (axis < -rank || axis >= rank) {
        throw InferenceException(context + ": axis out of range");
    }
    if (axis < 0) {
        axis += rank;
    }
    const int64_t exclusive = IntAttribute(node, "exclusive", 0, context);
    if (exclusive != 0 && exclusive != 1) {
        throw InferenceException(context + ": exclusive must be 0 or 1 here");
    }
    const int64_t reverse = IntAttribute(node, "reverse", 0, context);
    if (reverse != 0 && reverse != 1) {
        throw InferenceException(context + ": reverse must be 0 or 1 here");
    }
    if (out.dims != in.dims) {
        throw InferenceException(context + ": output shape must match the input");
    }
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
    const int64_t axisDim = in.dims[static_cast<std::size_t>(axis)];
    const bool excl = exclusive != 0;
    const bool rev = reverse != 0;
    if (in.dtype == DataType::Float32) {
        LaunchCumSum<float, double>(static_cast<const float*>(in.data), static_cast<float*>(out.data), axisDim, outer,
                                    inner, excl, rev, context);
    } else {
        LaunchCumSum<int64_t, int64_t>(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), axisDim,
                                       outer, inner, excl, rev, context);
    }
}

} // namespace cuda_kernels
} // namespace nert
