// EyeLike: 2-D output with the input shape, ones on the k-th diagonal and
// zeros elsewhere. `k` (default 0) offsets the diagonal upward (positive) or
// downward (negative). Data is float32, int64, or bool; the output dtype is
// fixed by shape inference from the `dtype` attribute or the input dtype.

#include "kernels.hpp"

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

} // namespace

void EyeLike(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "EyeLike (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dims.size() != 2 || out.dims != in.dims) {
        throw InferenceException(context + ": input and output must share a rank-2 shape");
    }
    if (out.dtype != DataType::Float32 && out.dtype != DataType::Int64 && out.dtype != DataType::Bool) {
        throw InferenceException(context + ": output must be float32, int64, or bool");
    }
    const int64_t k = IntAttribute(node, "k", 0, context);
    const int64_t rows = in.dims[0];
    const int64_t cols = in.dims[1];
    (void)ElementCount(out.dims, context);
    if (out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    if (out.dtype == DataType::Float32) {
        float* y = static_cast<float*>(out.data);
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                y[r * cols + c] = (c - r == k) ? 1.0F : 0.0F;
            }
        }
    } else if (out.dtype == DataType::Int64) {
        int64_t* y = static_cast<int64_t*>(out.data);
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                y[r * cols + c] = (c - r == k) ? 1 : 0;
            }
        }
    } else {
        bool* y = static_cast<bool*>(out.data);
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                y[r * cols + c] = (c - r == k);
            }
        }
    }
}

} // namespace kernels
} // namespace nert
