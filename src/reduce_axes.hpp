#pragma once
// Shared reduce-axes form resolution for ReduceMean, ReduceL2, and ReduceMax.
//
// ONNX schema 18 moved these operators' axes from an optional int64-list
// attribute to an optional int64 input. A node decoded from a model carries
// the effective schema in `Node.schema_version`, so consumers select the form
// here once at entry instead of rewriting the graph. Hand-built nodes (and
// nodes synthesized by a remaining load-time rewrite) have schema_version 0
// and keep the current input form.
//
// The header has no backend dependency: CPU, CUDA, OpenCL, and shape
// inference all read the same decision. Callers own the value extraction
// (an attribute value here, a device tensor for CUDA/OpenCL) and the empty
// axes semantics:
//   - input form: absent input means reduce all; a present empty input means
//     identity when `noop_with_empty_axes` is set. ReduceMean/ReduceL2
//     implement that identity; ReduceMax schema 18+ defines the attribute
//     but NERT rejects it as a tracked capability gap and otherwise reduces
//     all axes;
//   - attribute form: an absent or empty attribute means reduce all.

#include "nert/graph.hpp"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace nert {

// The schema version that introduced the optional axes input form.
constexpr int64_t kReduceAxesInputSchema = 18;

enum class ReduceAxesForm {
    Input,     // optional second int64 input: schema 18+, or schema_version 0
    Attribute, // optional int64-list `axes` attribute: source schema < 18
};

struct ReduceAxes {
    ReduceAxesForm form = ReduceAxesForm::Input;
    // The node's `axes` attribute when present (attribute form), else null.
    const Attribute* axes_attribute = nullptr;
    bool has_noop_with_empty_axes = false;
    bool noop_with_empty_axes = false;
};

// Returns the node attribute with `name`, or null when absent.
inline const Attribute* FindReduceAttribute(const Node& node, const std::string& name) {
    for (const Attribute& attribute : node.attributes) {
        if (attribute.name == name) {
            return &attribute;
        }
    }
    return nullptr;
}

// Returns the attribute's integer-list value, or null when the attribute is
// absent or carries another value kind.
inline const std::vector<int64_t>* ReduceAttributeIntList(const Attribute* attribute) {
    if (attribute == nullptr) {
        return nullptr;
    }
    return std::get_if<std::vector<int64_t>>(&attribute->value);
}

// Element count for an axes tensor. Unlike a data-tensor count, a zero
// dimension is valid here: it is ONNX's present-but-empty axes operand and
// must reach the noop/empty-axes logic rather than fail as a bad dimension.
inline int64_t ReduceAxesElementCount(const std::vector<int64_t>& dims, const std::string& context) {
    int64_t count = 1;
    for (const int64_t dim : dims) {
        if (dim < 0) {
            throw InferenceException(context + ": negative axes dimension");
        }
        if (dim == 0) {
            return 0;
        }
        count *= dim;
    }
    return count;
}

// Resolves the axes form and the noop flag from the node's effective schema.
// The caller decides whether an input tensor is present from the runtime
// input list (a hand-built test node may not record node.inputs).
inline ReduceAxes ResolveReduceAxes(const Node& node) {
    ReduceAxes resolved;
    resolved.form = node.schema_version != 0 && node.schema_version < kReduceAxesInputSchema ? ReduceAxesForm::Attribute
                                                                                             : ReduceAxesForm::Input;
    resolved.axes_attribute = FindReduceAttribute(node, "axes");
    const Attribute* noop = FindReduceAttribute(node, "noop_with_empty_axes");
    if (noop != nullptr) {
        resolved.has_noop_with_empty_axes = true;
        if (const int64_t* value = std::get_if<int64_t>(&noop->value)) {
            resolved.noop_with_empty_axes = *value != 0;
        }
    }
    return resolved;
}

} // namespace nert
