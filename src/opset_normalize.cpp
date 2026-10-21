// Load-time canonicalization of older ONNX node forms into the opset-18 IR.
// See opset_normalize.hpp.

#include "opset_normalize.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace nert {
namespace {

constexpr int64_t kAxesAsInputOpset = 18;
constexpr int64_t kScatterReductionOpset = 16;

std::string node_label(const Node& node) {
    return "node " + (node.name.empty() ? std::string("<unnamed>") : "'" + node.name + "'") + " op_type '" +
           node.op_type + "'";
}

const Attribute* find_attribute(const Node& node, const std::string& name) {
    for (const Attribute& attribute : node.attributes) {
        if (attribute.name == name) {
            return &attribute;
        }
    }
    return nullptr;
}

void erase_attribute(Node& node, const std::string& name) {
    for (auto it = node.attributes.begin(); it != node.attributes.end(); ++it) {
        if (it->name == name) {
            node.attributes.erase(it);
            return;
        }
    }
}

bool name_in_use(const Graph& graph, const std::string& name) {
    if (graph.initializers.count(name) != 0 || graph.tensors.count(name) != 0) {
        return true;
    }
    for (const auto& input : graph.inputs) {
        if (input.first == name) {
            return true;
        }
    }
    for (const auto& output : graph.outputs) {
        if (output.first == name) {
            return true;
        }
    }
    for (const Node& node : graph.nodes) {
        for (const std::string& input : node.inputs) {
            if (input == name) {
                return true;
            }
        }
        for (const std::string& output : node.outputs) {
            if (output == name) {
                return true;
            }
        }
    }
    return false;
}

// Stable, collision-free name for the synthesized axes initializer.
std::string fresh_axes_name(const Graph& graph, const Node& node) {
    std::string base = node.name.empty() ? node.op_type : node.name;
    base += "/axes";
    if (!name_in_use(graph, base)) {
        return base;
    }
    for (int64_t suffix = 1;; ++suffix) {
        const std::string candidate = base + "_" + std::to_string(suffix);
        if (!name_in_use(graph, candidate)) {
            return candidate;
        }
    }
}

// Opsets 13..17 keep the reduce axes in an attribute; opset 18 made them an
// optional second input. The canonical graph keeps the 18 form.
void canonicalize_reduce_axes(Graph& graph, Node& node, int64_t declared_opset) {
    const std::string who = node_label(node);
    const Attribute* axes_attribute = find_attribute(node, "axes");

    if (declared_opset >= kAxesAsInputOpset) {
        if (axes_attribute != nullptr) {
            throw ModelLoadException(who + ": 'axes' must be an input at opset " + std::to_string(declared_opset) +
                                     ", not an attribute");
        }
        return;
    }

    if (node.inputs.size() > 1) {
        throw ModelLoadException(who +
                                 ": the 'axes' input is only valid from opset 18; this model declares "
                                 "opset " +
                                 std::to_string(declared_opset));
    }
    if (find_attribute(node, "noop_with_empty_axes") != nullptr) {
        throw ModelLoadException(who +
                                 ": 'noop_with_empty_axes' is only valid from opset 18; this model "
                                 "declares opset " +
                                 std::to_string(declared_opset));
    }
    if (axes_attribute == nullptr) {
        return;
    }

    const std::vector<int64_t>* axes = std::get_if<std::vector<int64_t>>(&axes_attribute->value);
    if (axes == nullptr) {
        throw ModelLoadException(who + ": 'axes' must be an integer list");
    }
    if (axes->empty()) {
        // An empty axes attribute means "reduce every axis"; in the 18 form
        // that is expressed by omitting the optional axes input.
        erase_attribute(node, "axes");
        return;
    }

    const std::string name = fresh_axes_name(graph, node);
    Initializer initializer;
    initializer.dtype = DataType::Int64;
    initializer.dims = {static_cast<int64_t>(axes->size())};
    initializer.values = *axes;
    graph.initializers.emplace(name, std::move(initializer));
    graph.tensors[name] = TensorInfo{DataType::Int64, {static_cast<int64_t>(axes->size())}};
    node.inputs.push_back(name);
    erase_attribute(node, "axes");
}

// ScatterND gained an optional `reduction` attribute at opset 16. The kernels
// implement the overwrite form only, which is `none` (also the default), so an
// explicit "none" is dropped and every other value is rejected.
void canonicalize_scatter_nd(Node& node, int64_t declared_opset) {
    const Attribute* reduction = find_attribute(node, "reduction");
    if (reduction == nullptr) {
        return;
    }
    const std::string who = node_label(node);
    if (declared_opset < kScatterReductionOpset) {
        throw ModelLoadException(who +
                                 ": 'reduction' is only valid from opset 16; this model declares "
                                 "opset " +
                                 std::to_string(declared_opset));
    }
    const std::string* value = std::get_if<std::string>(&reduction->value);
    if (value == nullptr) {
        throw ModelLoadException(who + ": 'reduction' must be a string");
    }
    if (*value == "none") {
        erase_attribute(node, "reduction");
        return;
    }
    throw ModelLoadException(who + ": ScatterND reduction '" + *value + "' is not implemented here (only 'none')");
}

} // namespace

void NormalizeGraphForOpset(Graph& graph, int64_t declared_opset) {
    for (Node& node : graph.nodes) {
        if (node.op_type == "ReduceMean" || node.op_type == "ReduceL2" || node.op_type == "ReduceMax") {
            canonicalize_reduce_axes(graph, node, declared_opset);
        } else if (node.op_type == "ScatterND") {
            canonicalize_scatter_nd(node, declared_opset);
        }
    }
}

} // namespace nert
