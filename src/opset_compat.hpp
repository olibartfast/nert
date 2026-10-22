#pragma once
// Per-node ONNX schema catalog and opset compatibility checks.
//
// The committed table in `opset_compat_table.inc` is generated from ONNX's
// full schema history. The resolver selects the effective schema for a decoded
// node as `max { since_version <= declared_opset }`. The gate applies to every
// declaration: historical schemas are accepted after the loader's form
// validation, and newer schemas are accepted only when their capability
// guards are satisfied. Must not include anything from the backend abstraction
// layer.

#include <cstdint>
#include <string>
#include <vector>

namespace nert {

enum class OpsetCompatibility {
    Historical,         // older than the opset-18 baseline; catalog only
    Equivalent,         // the opset-18 implementation matches this schema
    AdditiveAttributes, // only attributes the implementation does not use
    ValueDomain,        // a new value in an existing attribute
    BehaviorChange,     // changed semantics or defaults
};

// The parts of a decoded attribute the compatibility gate needs: its name and,
// for value-domain guards, a string value. Unknown attributes are carried
// through this view before the strict per-operator allowlist runs.
struct OpsetAttributeView {
    std::string name;
    bool is_string = false;
    std::string string_value;
};

// Resolves the effective ONNX schema for a source node decoded from the model:
// the latest catalog entry for `op_type` whose schema version is at or below
// `declared_opset`. `node_name` is used only in diagnostics. Throws
// ModelLoadException naming the node, operator, and declaration when no schema
// entry exists at or below the declaration.
int64_t ResolveNodeSchemaVersion(const std::string& node_name, const std::string& op_type, int64_t declared_opset);

// Rejects a node whose effective schema version (the latest schema at or
// before `declared_opset`) is not executable by the implementation. Applies
// to every declaration; historical rows are accepted after the loader's form
// validation, and newer rows apply their attribute/value guards.
void CheckNodeCompatibility(const std::string& node_name, const std::string& op_type,
                            const std::vector<OpsetAttributeView>& attributes, int64_t declared_opset);

} // namespace nert
