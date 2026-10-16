// ArgMax: index of the maximum element along `axis` (default 0).
//
// `keepdims` (default 1) keeps the reduced axis as size 1, otherwise it is
// dropped. `select_last_index` (default 0) picks the last tied maximum
// instead of the first. Data is float32 or int64; indices are int64.

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

template <typename T>
void ArgMaxTyped(const T* x, int64_t* y, int64_t axis, int64_t dim, int64_t outer, int64_t inner, bool last) {
    for (int64_t o = 0; o < outer; ++o) {
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
        ArgMaxTyped(static_cast<const float*>(in.data), y, axis, dim, outer, inner, lastIndex);
    } else {
        ArgMaxTyped(static_cast<const int64_t*>(in.data), y, axis, dim, outer, inner, lastIndex);
    }
}

} // namespace kernels
} // namespace nert
