// Load-time canonicalization of the remaining older ONNX node forms.
// See opset_normalize.hpp.
//
// The version-aware operators (ReduceSum/Squeeze/Unsqueeze axes, ScatterND
// reduction) keep their source form in the graph; the loader gates legality
// and each consumer resolves the form from the node's effective schema.

#include "opset_normalize.hpp"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace nert {
namespace {

constexpr int64_t kSoftmaxCoercionOpset = 13;

std::string node_label(const Node& node) {
    return "node " + (node.name.empty() ? std::string("<unnamed>") : "'" + node.name + "'") + " op_type '" +
           node.op_type + "'";
}

std::string node_base(const Node& node) { return node.name.empty() ? node.op_type : node.name; }

const Attribute* find_attribute(const Node& node, const std::string& name) {
    for (const Attribute& attribute : node.attributes) {
        if (attribute.name == name) {
            return &attribute;
        }
    }
    return nullptr;
}

int64_t int_attribute_or(const Node& node, const std::string& name, int64_t fallback, const std::string& who) {
    const Attribute* attribute = find_attribute(node, name);
    if (attribute == nullptr) {
        return fallback;
    }
    const int64_t* value = std::get_if<int64_t>(&attribute->value);
    if (value == nullptr) {
        throw ModelLoadException(who + ": '" + name + "' must be an integer");
    }
    return *value;
}

// Stable, collision-free names for synthesized nodes, tensors, and
// initializers. One namespace covers both node and tensor names so generated
// names are unambiguous in diagnostics.
class NameAllocator {
  public:
    explicit NameAllocator(const Graph& graph) {
        for (const auto& input : graph.inputs) {
            reserve(input.first);
        }
        for (const auto& output : graph.outputs) {
            reserve(output.first);
        }
        for (const auto& entry : graph.tensors) {
            reserve(entry.first);
        }
        for (const auto& entry : graph.initializers) {
            reserve(entry.first);
        }
        for (const Node& node : graph.nodes) {
            reserve(node.name);
            for (const std::string& input : node.inputs) {
                reserve(input);
            }
            for (const std::string& output : node.outputs) {
                reserve(output);
            }
        }
    }

    std::string Fresh(const std::string& base) {
        std::string candidate = base;
        for (int64_t suffix = 1; used_.count(candidate) != 0; ++suffix) {
            candidate = base + "_" + std::to_string(suffix);
        }
        used_.insert(candidate);
        return candidate;
    }

  private:
    void reserve(const std::string& name) {
        if (!name.empty()) {
            used_.insert(name);
        }
    }

    std::set<std::string> used_;
};

const std::vector<int64_t>* known_dims(const Graph& graph, const std::string& name) {
    const auto initializer = graph.initializers.find(name);
    if (initializer != graph.initializers.end()) {
        return &initializer->second.dims;
    }
    for (const auto& input : graph.inputs) {
        if (input.first == name) {
            return &input.second.dims;
        }
    }
    const auto tensor = graph.tensors.find(name);
    if (tensor != graph.tensors.end()) {
        return &tensor->second.dims;
    }
    for (const auto& output : graph.outputs) {
        if (output.first == name) {
            return &output.second.dims;
        }
    }
    return nullptr;
}

std::optional<int64_t> known_rank(const Graph& graph, const std::string& name) {
    const std::vector<int64_t>* dims = known_dims(graph, name);
    if (dims == nullptr || dims->empty()) {
        return std::nullopt;
    }
    return static_cast<int64_t>(dims->size());
}

// Softmax before opset 13 coerced the input to 2D at `axis` and normalized
// over the second dimension. Opset 13 changed both the default (1 -> -1) and
// the semantics to a single-axis softmax. Preserve the old behavior with
// Shape -> Flatten(axis) -> Softmax(axis=1) -> Reshape.
void rewrite_legacy_softmax(const Graph& graph, NameAllocator& names, Node& node, int64_t declared_opset,
                            std::vector<Node>& out) {
    const std::string who = node_label(node);
    if (node.inputs.size() != 1 || node.inputs[0].empty()) {
        throw ModelLoadException(who + ": expected exactly one data input");
    }
    if (node.outputs.size() != 1 || node.outputs[0].empty()) {
        throw ModelLoadException(who + ": expected exactly one output");
    }

    const int64_t axis = int_attribute_or(node, "axis", 1, who);
    const std::optional<int64_t> rank = known_rank(graph, node.inputs[0]);
    if (!rank.has_value() || *rank <= 0) {
        throw ModelLoadException(who + ": legacy Softmax at opset " + std::to_string(declared_opset) +
                                 " needs a statically known, non-scalar rank for input '" + node.inputs[0] +
                                 "'; add value_info or convert the model to opset 13+");
    }
    int64_t normalized = axis;
    if (normalized < 0) {
        normalized += *rank;
    }
    if (normalized < 0 || normalized >= *rank) {
        throw ModelLoadException(who + ": 'axis' " + std::to_string(axis) + " is out of range for rank " +
                                 std::to_string(*rank) + " at opset " + std::to_string(declared_opset));
    }

    const std::string base = node_base(node);
    const std::string shape_name = names.Fresh(base + "/shape");
    const std::string shape_out = names.Fresh(base + "/shape_out");
    const std::string flatten_name = names.Fresh(base + "/flatten");
    const std::string flatten_out = names.Fresh(base + "/flatten_out");
    const std::string softmax_name = names.Fresh(base + "/softmax");
    const std::string softmax_out = names.Fresh(base + "/softmax_out");
    const std::string reshape_name = names.Fresh(base + "/reshape");

    Node shape;
    shape.name = shape_name;
    shape.op_type = "Shape";
    shape.inputs = {node.inputs[0]};
    shape.outputs = {shape_out};

    Node flatten;
    flatten.name = flatten_name;
    flatten.op_type = "Flatten";
    flatten.inputs = {node.inputs[0]};
    flatten.outputs = {flatten_out};
    flatten.attributes.push_back(Attribute("axis", normalized));

    Node softmax;
    softmax.name = softmax_name;
    softmax.op_type = "Softmax";
    softmax.inputs = {flatten_out};
    softmax.outputs = {softmax_out};
    softmax.attributes.push_back(Attribute("axis", int64_t{1}));

    Node reshape;
    reshape.name = reshape_name;
    reshape.op_type = "Reshape";
    reshape.inputs = {softmax_out, shape_out};
    reshape.outputs = {node.outputs[0]};

    out.push_back(std::move(shape));
    out.push_back(std::move(flatten));
    out.push_back(std::move(softmax));
    out.push_back(std::move(reshape));
}

} // namespace

void NormalizeGraphForOpset(Graph& graph, int64_t declared_opset) {
    NameAllocator names(graph);
    std::vector<Node> rewritten;
    rewritten.reserve(graph.nodes.size());
    for (Node& node : graph.nodes) {
        if (node.op_type == "Softmax" && declared_opset < kSoftmaxCoercionOpset) {
            rewrite_legacy_softmax(graph, names, node, declared_opset, rewritten);
        } else {
            rewritten.push_back(std::move(node));
        }
    }
    graph.nodes = std::move(rewritten);
}

} // namespace nert
