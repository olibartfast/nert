// Hipified from src/cuda/range.cu via hipify-perl, hand-reviewed.
// Range on HIP: 1-D sequence start, start + delta, ... up to (excluding)
// limit; float32 or int64.
//
// Mirrors the CPU oracle: scalar start/limit/delta downloaded to the host
// for length validation, then one thread per element fills. The float and
// int64 length rules match the oracle exactly.

#include "nert/device.hpp"

#include <cmath>
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

template <typename T> __global__ void RangeKernel(T* y, int64_t count, T start, T delta) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        y[i] = start + static_cast<T>(i) * delta;
    }
}

template <typename T> void LaunchRange(T* y, int64_t count, T start, T delta, const std::string& context) {
    if (count == 0) {
        return;
    }
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    RangeKernel<<<grid, block>>>(y, count, start, delta);
    CheckHip(hipGetLastError(), "RangeKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

template <typename T>
T ReadScalar(const TensorView& view, DataType dtype, const char* what, const std::string& context) {
    if (view.dtype != dtype) {
        throw InferenceException(context + ": " + std::string(what) + " dtype must match the other operands");
    }
    int64_t count = 1;
    for (int64_t dim : view.dims) {
        count *= dim;
    }
    if (count != 1) {
        throw InferenceException(context + ": " + std::string(what) + " must be a scalar");
    }
    if (view.data == nullptr) {
        throw InferenceException(context + ": null " + std::string(what) + " buffer");
    }
    T value{};
    CheckHip(hipMemcpy(&value, view.data, sizeof(T), hipMemcpyDeviceToHost), "hipMemcpy", context);
    return value;
}

} // namespace

void Range(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Range (node '" + node.name + "')";
    if (inputs.size() != 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected start, limit, delta and one output");
    }
    const TensorView& out = outputs[0];
    const DataType dtype = inputs[0].dtype;
    if ((dtype != DataType::Float32 && dtype != DataType::Int64) || out.dtype != dtype) {
        throw InferenceException(context + ": tensors must share dtype float32 or int64");
    }
    if (out.dims.size() != 1 || out.dims[0] < 0) {
        throw InferenceException(context + ": output must be rank 1");
    }
    const int64_t outCount = out.dims[0];

    if (dtype == DataType::Float32) {
        const float start = ReadScalar<float>(inputs[0], dtype, "start", context);
        const float limit = ReadScalar<float>(inputs[1], dtype, "limit", context);
        const float delta = ReadScalar<float>(inputs[2], dtype, "delta", context);
        if (delta == 0.0F) {
            throw InferenceException(context + ": delta must be non-zero");
        }
        const double span =
            std::ceil((static_cast<double>(limit) - static_cast<double>(start)) / static_cast<double>(delta));
        const int64_t expected = span > 0.0 ? static_cast<int64_t>(span) : 0;
        if (outCount != expected) {
            throw InferenceException(context + ": output length does not match the range");
        }
        if (outCount > 0 && out.data == nullptr) {
            throw InferenceException(context + ": null output buffer");
        }
        LaunchRange(static_cast<float*>(out.data), outCount, start, delta, context);
        return;
    }

    const int64_t start = ReadScalar<int64_t>(inputs[0], dtype, "start", context);
    const int64_t limit = ReadScalar<int64_t>(inputs[1], dtype, "limit", context);
    const int64_t delta = ReadScalar<int64_t>(inputs[2], dtype, "delta", context);
    if (delta == 0) {
        throw InferenceException(context + ": delta must be non-zero");
    }
    int64_t expected = 0;
    if (delta > 0 && limit > start) {
        expected = (limit - start + delta - 1) / delta;
    } else if (delta < 0 && limit < start) {
        expected = (start - limit + (-delta) - 1) / (-delta);
    }
    if (outCount != expected) {
        throw InferenceException(context + ": output length does not match the range");
    }
    if (outCount > 0 && out.data == nullptr) {
        throw InferenceException(context + ": null output buffer");
    }
    LaunchRange(static_cast<int64_t*>(out.data), outCount, start, delta, context);
}

} // namespace hip_kernels
} // namespace nert
