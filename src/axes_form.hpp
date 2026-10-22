#pragma once
// Shared axes-form resolution for ReduceSum, Squeeze, and Unsqueeze, plus the
// ScatterND reduction boundary.
//
// ONNX schema 13 moved these operators' axes from an int64-list attribute to
// an int64 input (a required input for Unsqueeze, optional for the others).
// A node decoded from a model carries the effective schema in
// `Node.schema_version`, so consumers select the source form here once at
// entry instead of rewriting the graph. Hand-built nodes (and nodes
// synthesized by a remaining load-time rewrite) have schema_version 0 and keep
// the current input form.
//
// The header has no backend dependency: CPU, CUDA, OpenCL, and shape
// inference all read the same decision, then extract the values themselves (an
// attribute list here, a device tensor for CUDA/OpenCL). Empty axes semantics
// stay with the caller:
//   - ReduceSum: an absent input/attribute means reduce every axis; a present
//     but empty input means identity only when noop_with_empty_axes is set;
//   - Squeeze: an absent or empty axes operand means every size-1 axis;
//   - Unsqueeze: axes are required and must not be empty.
//
// ScatterND gained an optional `reduction` attribute at schema 16. The kernels
// implement the overwrite form only, which is `none` (the default), so the
// attribute stays in the graph and only `none`/absent is legal from 16.

#include "nert/graph.hpp"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace nert {

// The schema version that introduced the axes-input form.
constexpr int64_t kAxesAsInputSchema = 13;

// The schema version that introduced ScatterND's `reduction` attribute.
constexpr int64_t kScatterReductionSchema = 16;

// Returns the node attribute with `name`, or null when absent.
inline const Attribute* FindNodeAttribute(const Node& node, const std::string& name) {
    for (const Attribute& attribute : node.attributes) {
        if (attribute.name == name) {
            return &attribute;
        }
    }
    return nullptr;
}

// Returns the attribute's integer-list value, or null when the attribute is
// absent or carries another value kind.
inline const std::vector<int64_t>* NodeAttributeIntList(const Attribute* attribute) {
    if (attribute == nullptr) {
        return nullptr;
    }
    return std::get_if<std::vector<int64_t>>(&attribute->value);
}

// Reads an integer attribute, returning `fallback` when absent. Throws
// InferenceException when the attribute is present with another value kind.
inline int64_t NodeAttributeIntOr(const Node& node, const std::string& name, int64_t fallback,
                                  const std::string& context) {
    const Attribute* attribute = FindNodeAttribute(node, name);
    if (attribute == nullptr) {
        return fallback;
    }
    const int64_t* value = std::get_if<int64_t>(&attribute->value);
    if (value == nullptr) {
        throw InferenceException(context + ": attribute '" + name + "' must be an integer");
    }
    return *value;
}

enum class AxesSource {
    Input,     // int64 axes operand: schema 13+, or schema_version 0
    Attribute, // int64-list `axes` attribute: source schema < 13
};

struct AxesForm {
    AxesSource source = AxesSource::Input;
    // The node's `axes` attribute when present, else null.
    const Attribute* axes_attribute = nullptr;
};

// Resolves the axes form from the node's effective schema.
inline AxesForm ResolveAxesForm(const Node& node) {
    AxesForm resolved;
    resolved.source = node.schema_version != 0 && node.schema_version < kAxesAsInputSchema ? AxesSource::Attribute
                                                                                           : AxesSource::Input;
    resolved.axes_attribute = FindNodeAttribute(node, "axes");
    return resolved;
}

// True when the node records a non-empty second operand. An empty-string
// operand is the exporter's omitted-input placeholder, not a tensor.
inline bool AxesInputPresent(const Node& node) { return node.inputs.size() > 1 && !node.inputs[1].empty(); }

// Element count for an axes tensor. Unlike a data-tensor count, a zero
// dimension is valid here: it is ONNX's present-but-empty axes operand and
// must reach the noop/empty-axes logic rather than fail as a bad dimension.
inline int64_t AxesElementCount(const std::vector<int64_t>& dims, const std::string& context) {
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

} // namespace nert
