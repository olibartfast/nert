// Load-time opset compatibility table lookup and guards.
// See opset_compat.hpp and tools/gen_opset_compat.py.

#include "opset_compat.hpp"

#include "nert/graph.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nert {
namespace {

struct OpsetSchemaInfo {
    const char* op_type;
    int64_t since_version;
    OpsetCompatibility compatibility;
    const char* added_attributes;
    const char* guarded_values;
    const char* reason;
};

static constexpr OpsetSchemaInfo kOpsetSchemaTable[] = {
#define NERT_OPSET_ENTRY(op, version, compatibility, added, guards, why)                                               \
    {op, version, OpsetCompatibility::compatibility, added, guards, why},
#include "opset_compat_table.inc"
#undef NERT_OPSET_ENTRY
};

constexpr size_t kOpsetSchemaTableSize = sizeof(kOpsetSchemaTable) / sizeof(kOpsetSchemaTable[0]);

std::string quote(const std::string& value) { return "'" + value + "'"; }

std::string node_prefix(const std::string& node_name, const std::string& op_type) {
    return "node " + (node_name.empty() ? std::string("<unnamed>") : quote(node_name)) + " op_type " + quote(op_type);
}

std::string version_text(int64_t declared_opset, int64_t effective_version) {
    return " (declared opset " + std::to_string(declared_opset) + ", effective schema " +
           std::to_string(effective_version) + ")";
}

std::vector<std::string> split_csv(const char* text) {
    std::vector<std::string> parts;
    if (text == nullptr || *text == '\0') {
        return parts;
    }
    std::string current;
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p == ',') {
            parts.push_back(current);
            current.clear();
        } else {
            current.push_back(*p);
        }
    }
    parts.push_back(current);
    return parts;
}

const OpsetAttributeView* find_attribute(const std::vector<OpsetAttributeView>& attributes, const std::string& name) {
    for (const OpsetAttributeView& attribute : attributes) {
        if (attribute.name == name) {
            return &attribute;
        }
    }
    return nullptr;
}

const OpsetSchemaInfo* find_effective_schema(const std::string& op_type, int64_t declared_opset) {
    const OpsetSchemaInfo* best = nullptr;
    for (size_t i = 0; i < kOpsetSchemaTableSize; ++i) {
        const OpsetSchemaInfo& entry = kOpsetSchemaTable[i];
        if (op_type != entry.op_type || entry.since_version > declared_opset) {
            continue;
        }
        if (best == nullptr || entry.since_version > best->since_version) {
            best = &entry;
        }
    }
    return best;
}

int64_t first_schema_version(const std::string& op_type) {
    int64_t first = -1;
    for (size_t i = 0; i < kOpsetSchemaTableSize; ++i) {
        const OpsetSchemaInfo& entry = kOpsetSchemaTable[i];
        if (op_type != entry.op_type) {
            continue;
        }
        if (first < 0 || entry.since_version < first) {
            first = entry.since_version;
        }
    }
    return first;
}

} // namespace

int64_t ResolveNodeSchemaVersion(const std::string& node_name, const std::string& op_type, int64_t declared_opset) {
    const OpsetSchemaInfo* schema = find_effective_schema(op_type, declared_opset);
    if (schema == nullptr) {
        const int64_t first = first_schema_version(op_type);
        const std::string detail = first < 0 ? std::string(" (operator is not in the schema catalog)")
                                             : " (operator's first schema is " + std::to_string(first) + ")";
        throw ModelLoadException(node_prefix(node_name, op_type) + ": no effective schema at or below declared opset " +
                                 std::to_string(declared_opset) + detail);
    }
    return schema->since_version;
}

void CheckNodeCompatibility(const std::string& node_name, const std::string& op_type,
                            const std::vector<OpsetAttributeView>& attributes, int64_t declared_opset) {
    const std::string prefix = node_prefix(node_name, op_type);
    const OpsetSchemaInfo* schema = find_effective_schema(op_type, declared_opset);
    if (schema == nullptr) {
        throw ModelLoadException(prefix +
                                 ": no opset compatibility entry for this operator (declared "
                                 "opset " +
                                 std::to_string(declared_opset) + ")");
    }
    const std::string versions = version_text(declared_opset, schema->since_version);

    if (schema->compatibility == OpsetCompatibility::Historical) {
        // Historical rows are catalog entries, not admission verdicts. The
        // above-18 gate should never select one (the baseline is also <=18);
        // fail closed if a future caller routes an old declaration here.
        throw ModelLoadException(prefix + ": historical schema " + std::to_string(schema->since_version) +
                                 " reached the compatibility gate" + versions);
    }

    if (schema->compatibility == OpsetCompatibility::BehaviorChange) {
        throw ModelLoadException(prefix + ": schema " + std::to_string(schema->since_version) + " " +
                                 (schema->reason == nullptr ? "changed semantics" : schema->reason) +
                                 "; this behavior is not implemented" + versions + " (tracked follow-up)");
    }

    for (const std::string& attribute : split_csv(schema->added_attributes)) {
        if (find_attribute(attributes, attribute) != nullptr) {
            throw ModelLoadException(prefix + ": attribute " + quote(attribute) + " is not implemented" + versions +
                                     " (tracked follow-up: " + op_type + "." + attribute + ")");
        }
    }

    for (const std::string& guard : split_csv(schema->guarded_values)) {
        const std::size_t equals = guard.find('=');
        if (equals == std::string::npos) {
            continue;
        }
        const std::string name = guard.substr(0, equals);
        const std::string value = guard.substr(equals + 1);
        const OpsetAttributeView* attribute = find_attribute(attributes, name);
        if (attribute != nullptr && attribute->is_string && attribute->string_value == value) {
            throw ModelLoadException(prefix + ": " + quote(name) + " value " + quote(value) + " is not implemented" +
                                     versions + " (tracked follow-up: " + op_type + "." + name + ")");
        }
    }
}

} // namespace nert
