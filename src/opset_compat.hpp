#pragma once
// Per-node ONNX schema catalog and implementation checks.
//
// The committed table in `opset_compat_table.inc` is generated from ONNX's
// full schema history and the reviewed implementation manifest in
// `tools/opset_manifest.py`. The resolver selects the effective schema for a
// decoded node as `max { since_version <= declared_opset }`. The gate applies
// to every declaration: a schema is accepted only when the runtime implements
// it, and any unimplemented attribute or value is rejected explicitly. There
// is no canonical baseline opset. Must not include anything from the backend
// abstraction layer.

#include <cstdint>
#include <string>
#include <vector>

namespace nert {

enum class OpsetCompatibility {
    Implemented,       // schema semantics and all its attributes are implemented
    GuardedAttributes, // listed attributes are not implemented and are rejected when present
    ValueDomain,       // listed attribute values are not implemented
    BehaviorChange,    // changed semantics or defaults; the schema is not implemented
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
// before `declared_opset`) is not executable by the implementation. Applies to
// every declaration. Guarded attributes and values fail only when the node
// uses them; a BehaviorChange schema fails for every node.
void CheckNodeCompatibility(const std::string& node_name, const std::string& op_type,
                            const std::vector<OpsetAttributeView>& attributes, int64_t declared_opset);

} // namespace nert
