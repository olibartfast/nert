// Load-time canonicalization of older ONNX node forms into the opset-18 IR.
// See opset_normalize.hpp.

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

constexpr int64_t kReduceAxesAsInputOpset = 18;
constexpr int64_t kScatterReductionOpset = 16;
constexpr int64_t kAxesAsInputOpset = 13;
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

void erase_attribute(Node& node, const std::string& name) {
    for (auto it = node.attributes.begin(); it != node.attributes.end(); ++it) {
        if (it->name == name) {
            node.attributes.erase(it);
            return;
        }
    }
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

std::vector<int64_t> int_list_attribute(const Attribute& attribute, const std::string& who) {
    const std::vector<int64_t>* values = std::get_if<std::vector<int64_t>>(&attribute.value);
    if (values == nullptr) {
        throw ModelLoadException(who + ": '" + attribute.name + "' must be an integer list");
    }
    return *values;
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

// Appends an Int64 initializer as the next node input and records it in the
// tensor table. The caller erases the old attribute.
void append_int64_input(Graph& graph, NameAllocator& names, Node& node, const std::vector<int64_t>& values,
                        const std::string& suffix) {
    const std::string name = names.Fresh(node_base(node) + "/" + suffix);
    Initializer initializer;
    initializer.dtype = DataType::Int64;
    initializer.dims = {static_cast<int64_t>(values.size())};
    initializer.values = values;
    graph.initializers.emplace(name, std::move(initializer));
    graph.tensors[name] = TensorInfo{DataType::Int64, {static_cast<int64_t>(values.size())}};
    node.inputs.push_back(name);
}

// Opsets 11..17 keep the reduce axes in an attribute; opset 18 made them an
// optional second input. The canonical graph keeps the 18 form.
void canonicalize_reduce_axes(Graph& graph, NameAllocator& names, Node& node, int64_t declared_opset) {
    const std::string who = node_label(node);
    const Attribute* axes_attribute = find_attribute(node, "axes");

    if (declared_opset >= kReduceAxesAsInputOpset) {
        if (axes_attribute != nullptr) {
            throw ModelLoadException(who + ": 'axes' must be an input at opset " + std::to_string(declared_opset) +
                                     ", not an attribute");
        }
        return;
    }

    if (node.inputs.size() > 1) {
        throw ModelLoadException(who + ": the 'axes' input is only valid from opset 18; this model declares opset " +
                                 std::to_string(declared_opset));
    }
    if (find_attribute(node, "noop_with_empty_axes") != nullptr) {
        throw ModelLoadException(who + ": 'noop_with_empty_axes' is only valid from opset 18; this model " +
                                 "declares opset " + std::to_string(declared_opset));
    }
    if (axes_attribute == nullptr) {
        return;
    }

    const std::vector<int64_t> axes = int_list_attribute(*axes_attribute, who);
    if (axes.empty()) {
        // An empty axes attribute means "reduce every axis"; in the 18 form
        // that is expressed by omitting the optional axes input.
        erase_attribute(node, "axes");
        return;
    }

    append_int64_input(graph, names, node, axes, "axes");
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
        throw ModelLoadException(who + ": 'reduction' is only valid from opset 16; this model declares " + "opset " +
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

// ReduceSum took axes as an input from opset 13. Before that it was an
// optional attribute; an empty list means reduce every axis, which is the
// absent-input form in the canonical graph.
void canonicalize_reduce_sum(Graph& graph, NameAllocator& names, Node& node, int64_t declared_opset) {
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
        throw ModelLoadException(who + ": the 'axes' input is only valid from opset 13; this model declares opset " +
                                 std::to_string(declared_opset));
    }
    if (find_attribute(node, "noop_with_empty_axes") != nullptr) {
        throw ModelLoadException(who + ": 'noop_with_empty_axes' is only valid from opset 13; this model " +
                                 "declares opset " + std::to_string(declared_opset));
    }
    if (axes_attribute == nullptr) {
        return;
    }

    const std::vector<int64_t> axes = int_list_attribute(*axes_attribute, who);
    if (axes.empty()) {
        erase_attribute(node, "axes");
        return;
    }
    append_int64_input(graph, names, node, axes, "axes");
    erase_attribute(node, "axes");
}

// Squeeze's axes moved from an attribute (11) to an optional input (13). An
// empty attribute list means "all size-1 axes", the same as omitting axes.
void canonicalize_squeeze(Graph& graph, NameAllocator& names, Node& node, int64_t declared_opset) {
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
        throw ModelLoadException(who + ": the 'axes' input is only valid from opset 13; this model declares opset " +
                                 std::to_string(declared_opset));
    }
    if (axes_attribute == nullptr) {
        return;
    }

    const std::vector<int64_t> axes = int_list_attribute(*axes_attribute, who);
    if (axes.empty()) {
        erase_attribute(node, "axes");
        return;
    }
    append_int64_input(graph, names, node, axes, "axes");
    erase_attribute(node, "axes");
}

// Unsqueeze's axes moved from a required attribute (11) to a required input
// (13). An empty attribute list is rejected: ONNX has no no-op form here.
void canonicalize_unsqueeze(Graph& graph, NameAllocator& names, Node& node, int64_t declared_opset) {
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
        throw ModelLoadException(who + ": the 'axes' input is only valid from opset 13; this model declares opset " +
                                 std::to_string(declared_opset));
    }
    if (axes_attribute == nullptr) {
        throw ModelLoadException(who + ": 'axes' is required at opset " + std::to_string(declared_opset));
    }

    const std::vector<int64_t> axes = int_list_attribute(*axes_attribute, who);
    if (axes.empty()) {
        throw ModelLoadException(who + ": 'axes' must not be empty at opset " + std::to_string(declared_opset));
    }
    append_int64_input(graph, names, node, axes, "axes");
    erase_attribute(node, "axes");
}

// Split's `split` moved from an attribute (11) to an optional input (13). An
// empty list cannot describe any output partition and is rejected.
void canonicalize_split(Graph& graph, NameAllocator& names, Node& node, int64_t declared_opset) {
    const std::string who = node_label(node);
    const Attribute* split_attribute = find_attribute(node, "split");

    if (declared_opset >= kAxesAsInputOpset) {
        if (split_attribute != nullptr) {
            throw ModelLoadException(who + ": 'split' must be an input at opset " + std::to_string(declared_opset) +
                                     ", not an attribute");
        }
        return;
    }
    if (node.inputs.size() > 1) {
        throw ModelLoadException(who + ": the 'split' input is only valid from opset 13; this model declares " +
                                 "opset " + std::to_string(declared_opset));
    }
    if (split_attribute == nullptr) {
        return;
    }

    const std::vector<int64_t> split = int_list_attribute(*split_attribute, who);
    if (split.empty()) {
        throw ModelLoadException(who + ": 'split' must not be empty at opset " + std::to_string(declared_opset));
    }
    append_int64_input(graph, names, node, split, "split");
    erase_attribute(node, "split");
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

void normalize_node(Graph& graph, NameAllocator& names, Node& node, int64_t declared_opset) {
    if (node.op_type == "ReduceMean" || node.op_type == "ReduceL2" || node.op_type == "ReduceMax") {
        canonicalize_reduce_axes(graph, names, node, declared_opset);
    } else if (node.op_type == "ReduceSum") {
        canonicalize_reduce_sum(graph, names, node, declared_opset);
    } else if (node.op_type == "Squeeze") {
        canonicalize_squeeze(graph, names, node, declared_opset);
    } else if (node.op_type == "Unsqueeze") {
        canonicalize_unsqueeze(graph, names, node, declared_opset);
    } else if (node.op_type == "Split") {
        canonicalize_split(graph, names, node, declared_opset);
    } else if (node.op_type == "ScatterND") {
        canonicalize_scatter_nd(node, declared_opset);
    }
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
            normalize_node(graph, names, node, declared_opset);
            rewritten.push_back(std::move(node));
        }
    }
    graph.nodes = std::move(rewritten);
}

} // namespace nert
