#pragma once
// Shared Softmax semantic-form resolution.
//
// ONNX schema 13 changed Softmax from a 2-D coercion at `axis` (default 1) to
// a single-axis normalization (default -1). A source node carries its
// effective schema in `Node.schema_version`; hand-built nodes use schema 0 and
// the current single-axis form. The device rows resolve the form once at entry
// and never branch on the version inside an element loop.

#include "nert/graph.hpp"

#include <cstdint>
#include <string>
#include <variant>

namespace nert {

constexpr int64_t kSoftmaxAxisSchema = 13;

enum class SoftmaxForm {
    Axis,          // schema 13+: normalize one axis, default -1
    LegacyCoerce2D // schema <13: flatten at axis (default 1), normalize dim 1
};

inline SoftmaxForm ResolveSoftmaxForm(const Node& node) {
    return node.schema_version != 0 && node.schema_version < kSoftmaxAxisSchema ? SoftmaxForm::LegacyCoerce2D
                                                                                : SoftmaxForm::Axis;
}

// Returns the source schema's default axis for the resolved form.
inline int64_t SoftmaxDefaultAxis(SoftmaxForm form) { return form == SoftmaxForm::LegacyCoerce2D ? 1 : -1; }

// Reads the `axis` attribute, or the form's default when absent.
inline int64_t SoftmaxAxis(const Node& node, SoftmaxForm form, const std::string& context) {
    for (const Attribute& attribute : node.attributes) {
        if (attribute.name != "axis") {
            continue;
        }
        const int64_t* value = std::get_if<int64_t>(&attribute.value);
        if (value == nullptr) {
            throw InferenceException(context + ": attribute 'axis' must be an integer");
        }
        return *value;
    }
    return SoftmaxDefaultAxis(form);
}

} // namespace nert
