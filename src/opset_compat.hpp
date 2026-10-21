#pragma once
// Per-node opset compatibility checks for declarations above opset 18.
//
// The committed table in `opset_compat_table.inc` is generated from ONNX's
// schema history. This seam only decides whether the opset-18 kernels can
// execute the node; it does not implement newer semantics. Must not include
// anything from the backend abstraction layer.

#include <cstdint>
#include <string>
#include <vector>

namespace nert {

enum class OpsetCompatibility {
    Equivalent,
    AdditiveAttributes,
    ValueDomain,
    BehaviorChange,
};

// The parts of a decoded attribute the compatibility gate needs: its name and,
// for value-domain guards, a string value. Unknown attributes are carried
// through this view before the strict per-operator allowlist runs.
struct OpsetAttributeView {
    std::string name;
    bool is_string = false;
    std::string string_value;
};

// Rejects a node whose effective schema version (the latest schema at or
// before `declared_opset`) is not executable by the opset-18 implementation.
// `declared_opset` must be above 18; older declarations use the range
// packet's canonicalizer instead.
void CheckNodeCompatibility(const std::string& node_name, const std::string& op_type,
                            const std::vector<OpsetAttributeView>& attributes, int64_t declared_opset);

} // namespace nert
