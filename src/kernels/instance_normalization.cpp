// InstanceNormalization: for each instance n and channel c of an [N, C, D1,
// ..., Dk] input, normalize the D1 * ... * Dk spatial elements to zero mean and
// unit variance (population variance plus `epsilon`, default 1e-5), then
// multiply by scale[c] and add B[c]. scale and B are rank-1 of length C. A
// rank-2 input has a single spatial element per (n, c). Statistics accumulate
// in double; data is float32.

#include "kernels.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace nert {
namespace kernels {
namespace {

float FloatAttribute(const Node& node, const std::string& name, float fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const float* value = std::get_if<float>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("InstanceNormalization: attribute '" + name + "' must be a float (node '" +
                                         node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

// A per-channel parameter (scale or B): rank-1 float32 of length `channels`.
const float* ChannelParam(const TensorView& view, int64_t channels, const std::string& context, const char* what) {
    if (view.dtype != DataType::Float32 || view.data == nullptr) {
        throw InferenceException(context + ": " + what + " must be a float32 tensor");
    }
    if (view.dims.size() != 1 || view.dims[0] != channels) {
        throw InferenceException(context + ": " + what + " must be rank-1 with one value per channel");
    }
    return static_cast<const float*>(view.data);
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
    const int64_t batch = x.dims[0];
    const int64_t channels = x.dims[1];
    int64_t spatial = 1;
    for (std::size_t i = 2; i < x.dims.size(); ++i) {
        spatial *= x.dims[i];
    }
    const float* scale = ChannelParam(inputs[1], channels, context, "scale");
    const float* bias = ChannelParam(inputs[2], channels, context, "B");
    const double epsilon = static_cast<double>(FloatAttribute(node, "epsilon", 1e-5F));

    const float* xData = static_cast<const float*>(x.data);
    float* yData = static_cast<float*>(y.data);
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t c = 0; c < channels; ++c) {
            const int64_t offset = (n * channels + c) * spatial;
            const float* in = xData + offset;
            float* out = yData + offset;
            double sum = 0.0;
            for (int64_t i = 0; i < spatial; ++i) {
                sum += in[i];
            }
            const double mean = sum / static_cast<double>(spatial);
            double squares = 0.0;
            for (int64_t i = 0; i < spatial; ++i) {
                const double centered = in[i] - mean;
                squares += centered * centered;
            }
            const double invStd = 1.0 / std::sqrt(squares / static_cast<double>(spatial) + epsilon);
            const double s = scale[c];
            const double b = bias[c];
            for (int64_t i = 0; i < spatial; ++i) {
                out[i] = static_cast<float>((in[i] - mean) * invStd * s + b);
            }
        }
    }
}

} // namespace kernels
} // namespace nert
