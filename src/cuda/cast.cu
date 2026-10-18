// Cast on CUDA: element conversion, one thread per element.
//
// Mirrors the CPU oracle: f32/i64/bool input, required int 'to' attr
// (1/6/7/9), output dtype/shape checks. Every conversion is a plain
// static_cast except to=6 (INT32), which is emulated on int64 lanes with
// portable saturation exactly like the CPU row (NaN becomes 0).

#include "nert/device.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace nert {
namespace cuda_kernels {
namespace {

constexpr int64_t kElemFloat = 1;
constexpr int64_t kElemInt32 = 6;
constexpr int64_t kElemInt64 = 7;
constexpr int64_t kElemBool = 9;

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

__device__ int64_t SaturateInt32Device(float value) {
    const double v = static_cast<double>(value);
    if (isnan(v)) {
        return 0;
    }
    if (v >= 2147483647.0) {
        return 2147483647LL;
    }
    if (v <= -2147483648.0) {
        return -2147483648LL;
    }
    return static_cast<int64_t>(v);
}

template <typename In, typename Out> __global__ void CastKernel(const In* x, Out* y, int64_t count) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        y[i] = static_cast<Out>(x[i]);
    }
}

__global__ void CastFloatToInt32Kernel(const float* x, int64_t* y, int64_t count) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        y[i] = SaturateInt32Device(x[i]);
    }
}

__global__ void CastInt64ToInt32Kernel(const int64_t* x, int64_t* y, int64_t count) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        const int64_t v = x[i];
        y[i] = v > 2147483647LL ? 2147483647LL : (v < -2147483648LL ? -2147483648LL : v);
    }
}

template <typename In, typename Out> void LaunchCast(const In* x, Out* y, int64_t count, const std::string& context) {
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    CastKernel<<<grid, block>>>(x, y, count);
    CheckCuda(cudaGetLastError(), "CastKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace

void Cast(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Cast (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool) {
        throw InferenceException(context + ": input must be float32, int64, or bool");
    }

    int64_t to = 0;
    bool have_to = false;
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "to") {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'to' must be an integer");
            }
            to = *value;
            have_to = true;
        }
    }
    if (!have_to) {
        throw InferenceException(context + ": missing required attribute 'to'");
    }
    DataType want = DataType::Unknown;
    if (to == kElemFloat) {
        want = DataType::Float32;
    } else if (to == kElemInt32) {
        // Emulated on int64 lanes (see file doc).
        want = DataType::Int64;
    } else if (to == kElemInt64) {
        want = DataType::Int64;
    } else if (to == kElemBool) {
        want = DataType::Bool;
    } else {
        throw InferenceException(context + ": attribute 'to' names an unsupported element type");
    }
    if (out.dtype != want) {
        throw InferenceException(context + ": output dtype does not match attribute 'to'");
    }

    const int64_t outCount = ElementCount(out.dims, context);
    const int64_t inCount = ElementCount(in.dims, context);
    if (inCount != outCount || in.dims != out.dims) {
        throw InferenceException(context + ": output shape must match the input shape");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (outCount == 0) {
        return;
    }

    const void* x = in.data;
    void* y = out.data;
    if (want == DataType::Float32) {
        auto* yf = static_cast<float*>(y);
        if (in.dtype == DataType::Float32) {
            LaunchCast(static_cast<const float*>(x), yf, outCount, context);
        } else if (in.dtype == DataType::Int64) {
            LaunchCast(static_cast<const int64_t*>(x), yf, outCount, context);
        } else {
            LaunchCast(static_cast<const bool*>(x), yf, outCount, context);
        }
    } else if (want == DataType::Int64) {
        auto* yi = static_cast<int64_t*>(y);
        if (to == kElemInt32 && in.dtype == DataType::Float32) {
            const int block = 256;
            const int grid = static_cast<int>((outCount + block - 1) / block);
            CastFloatToInt32Kernel<<<grid, block>>>(static_cast<const float*>(x), yi, outCount);
            CheckCuda(cudaGetLastError(), "CastFloatToInt32Kernel launch", context);
            CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
        } else if (to == kElemInt32 && in.dtype == DataType::Int64) {
            const int block = 256;
            const int grid = static_cast<int>((outCount + block - 1) / block);
            CastInt64ToInt32Kernel<<<grid, block>>>(static_cast<const int64_t*>(x), yi, outCount);
            CheckCuda(cudaGetLastError(), "CastInt64ToInt32Kernel launch", context);
            CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
        } else if (in.dtype == DataType::Float32) {
            LaunchCast(static_cast<const float*>(x), yi, outCount, context);
        } else if (in.dtype == DataType::Int64) {
            LaunchCast(static_cast<const int64_t*>(x), yi, outCount, context);
        } else {
            LaunchCast(static_cast<const bool*>(x), yi, outCount, context);
        }
    } else {
        auto* yb = static_cast<bool*>(y);
        if (in.dtype == DataType::Float32) {
            LaunchCast(static_cast<const float*>(x), yb, outCount, context);
        } else if (in.dtype == DataType::Int64) {
            LaunchCast(static_cast<const int64_t*>(x), yb, outCount, context);
        } else {
            LaunchCast(static_cast<const bool*>(x), yb, outCount, context);
        }
    }
}

} // namespace cuda_kernels
} // namespace nert
