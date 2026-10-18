// InstanceNormalization on CUDA: per-(n, c) normalization over the
// spatial elements.
//
// Mirrors the CPU oracle: float32 data, rank-1 [C] scale and B, epsilon
// default 1e-5, statistics in double. One thread per (n, c) row of
// `spatial` elements (the LayerNormalization pattern with per-channel
// parameters).

#include "nert/device.hpp"

#include <cmath>
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

float FloatAttribute(const Node& node, const std::string& name, float fallback, const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const float* value = std::get_if<float>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute '" + name + "' must be a float");
            }
            return *value;
        }
    }
    return fallback;
}

// A per-channel parameter (scale or B) on the device: rank-1 float32 of
// length `channels`.
const float* ChannelParam(const TensorView& view, int64_t channels, const std::string& context, const char* what) {
    if (view.dtype != DataType::Float32 || view.data == nullptr) {
        throw InferenceException(context + ": " + std::string(what) + " must be a float32 tensor");
    }
    if (view.dims.size() != 1 || view.dims[0] != channels) {
        throw InferenceException(context + ": " + std::string(what) + " must be rank-1 with one value per channel");
    }
    return static_cast<const float*>(view.data);
}

__global__ void InstanceNormalizationKernel(const float* x, float* y, const float* scale, const float* bias,
                                            int64_t rows, int64_t channels, int64_t spatial, double epsilon) {
    for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += blockDim.x * gridDim.x) {
        const float* row = x + r * spatial;
        float* outRow = y + r * spatial;
        double sum = 0.0;
        for (int64_t i = 0; i < spatial; ++i) {
            sum += row[i];
        }
        const double mean = sum / static_cast<double>(spatial);
        double squares = 0.0;
        for (int64_t i = 0; i < spatial; ++i) {
            const double centered = row[i] - mean;
            squares += centered * centered;
        }
        const double invStd = 1.0 / sqrt(squares / static_cast<double>(spatial) + epsilon);
        // Rows are ordered (n, c): the channel of row r is r % channels.
        const float s = scale[r % channels];
        const float b = bias[r % channels];
        for (int64_t i = 0; i < spatial; ++i) {
            outRow[i] = static_cast<float>((row[i] - mean) * invStd * s + b);
        }
    }
}

} // namespace

void InstanceNormalization(const Node& node, const std::vector<TensorView>& inputs,
                           const std::vector<TensorView>& outputs) {
    const std::string context = "InstanceNormalization (node '" + node.name + "')";
    if (inputs.size() != 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected input, scale, B, and one output");
    }
    const TensorView& x = inputs[0];
    const TensorView& y = outputs[0];
    if (x.dtype != DataType::Float32 || y.dtype != DataType::Float32) {
        throw InferenceException(context + ": input and output must be float32");
    }
    if (x.dims.size() < 2 || x.dims != y.dims) {
        throw InferenceException(context + ": output must have the shape of an input of rank >= 2");
    }
    for (int64_t dim : x.dims) {
        if (dim <= 0) {
            throw InferenceException(context + ": non-positive dimension");
        }
    }
    if (x.data == nullptr || y.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    const int64_t channels = x.dims[1];
    int64_t spatial = 1;
    for (std::size_t i = 2; i < x.dims.size(); ++i) {
        spatial *= x.dims[i];
    }
    const float* scale = ChannelParam(inputs[1], channels, context, "scale");
    const float* bias = ChannelParam(inputs[2], channels, context, "B");
    const double epsilon = static_cast<double>(FloatAttribute(node, "epsilon", 1e-5F, context));

    const int64_t batch = x.dims[0];
    const int64_t rows = batch * channels;
    if (rows == 0) {
        return;
    }
    const int block = 256;
    const int grid = static_cast<int>((rows + block - 1) / block);
    InstanceNormalizationKernel<<<grid, block>>>(static_cast<const float*>(x.data), static_cast<float*>(y.data), scale,
                                                 bias, rows, channels, spatial, epsilon);
    CheckCuda(cudaGetLastError(), "InstanceNormalizationKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace cuda_kernels
} // namespace nert
