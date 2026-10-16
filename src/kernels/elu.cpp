// Elu: elementwise x >= 0 ? x : alpha * (exp(x) - 1), `alpha` defaults to 1.0.

#include "kernels.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace nert {
namespace kernels {
namespace {

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

float FloatAttribute(const Node& node, const std::string& name, float fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const float* value = std::get_if<float>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("Elu: attribute '" + name + "' must be a float (node '" + node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

} // namespace

void Elu(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Elu (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }

    const int64_t outCount = ElementCount(out.dims, context);
    const int64_t inCount = ElementCount(in.dims, context);
    if (inCount != outCount) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const double alpha = static_cast<double>(FloatAttribute(node, "alpha", 1.0F));
    const float* x = static_cast<const float*>(in.data);
    float* y = static_cast<float*>(out.data);
    for (int64_t i = 0; i < outCount; ++i) {
        const double v = static_cast<double>(x[i]);
        y[i] = v >= 0.0 ? x[i] : static_cast<float>(alpha * (std::exp(v) - 1.0));
    }
}

} // namespace kernels
} // namespace nert
