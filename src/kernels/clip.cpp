// Clip: elementwise clamp of float32 or int64 data to optional scalar min and
// max bounds.

#include "kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
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

// Reads an optional scalar bound; returns `fallback` when the input is omitted
// (position past the end or an Unknown-dtype slot). A present bound must carry
// the data dtype, hold exactly one element, and have a live buffer.
template <typename T>
T ReadBound(const std::vector<TensorView>& inputs, size_t index, T fallback, DataType dtype, const char* type_name,
            const std::string& context) {
    if (index >= inputs.size() || inputs[index].dtype == DataType::Unknown) {
        return fallback;
    }
    const TensorView& bound = inputs[index];
    if (bound.dtype != dtype) {
        throw InferenceException(context + ": min/max must be " + type_name);
    }
    if (ElementCount(bound.dims, context) != 1) {
        throw InferenceException(context + ": min/max must be scalar");
    }
    if (bound.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    return *static_cast<const T*>(bound.data);
}

// Shared clamp loop for one dtype. Integer paths stay on integer comparisons
// (no float round-trip), and the absent-bound fallbacks make the clamp an
// exact identity: min(max(v, T_min), T_max) == v.
template <typename T>
void RunClip(const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs, const std::string& context,
             T lo_fallback, T hi_fallback, DataType dtype, const char* type_name) {
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    const int64_t count = ElementCount(out.dims, context);
    if (ElementCount(in.dims, context) != count) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    const T lo = ReadBound(inputs, 1, lo_fallback, dtype, type_name, context);
    const T hi = ReadBound(inputs, 2, hi_fallback, dtype, type_name, context);

    const T* x = static_cast<const T*>(in.data);
    T* y = static_cast<T*>(out.data);
    for (int64_t i = 0; i < count; ++i) {
        y[i] = std::min(std::max(x[i], lo), hi);
    }
}

} // namespace

void Clip(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Clip (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one to three inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != out.dtype) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (in.dtype == DataType::Float32) {
        RunClip<float>(inputs, outputs, context, -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::infinity(), DataType::Float32, "float32");
    } else if (in.dtype == DataType::Int64) {
        RunClip<int64_t>(inputs, outputs, context, std::numeric_limits<int64_t>::min(),
                         std::numeric_limits<int64_t>::max(), DataType::Int64, "int64");
    } else {
        throw InferenceException(context + ": tensors must be float32");
    }
}

} // namespace kernels
} // namespace nert
