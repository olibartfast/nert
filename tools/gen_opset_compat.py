#!/usr/bin/env python3
"""Generate the committed per-operator opset compatibility table.

Development-only tool. The runtime never imports ONNX; it compiles the
generated `src/opset_compat_table.inc`.

Usage:
    python3 tools/gen_opset_compat.py          # rewrite the table
    python3 tools/gen_opset_compat.py --check  # fail if the table is stale

The table covers every operator in `src/kernels/cpu_kernels.cpp`. For each
operator it records every schema version in ONNX's history. Versions before
the latest schema at or before opset 18 (the canonical implementation level)
are labeled `Historical`: the resolver can select them for older declarations,
but they are not claimed equivalent to the opset-18 implementation. The
baseline entry is `Equivalent`, and every later schema version carries a
compatibility class relative to the opset-18 implementation.

Phase A is transitional: pre-baseline rows carry only their version, not
reviewed attribute or value-domain deltas. Phase B/C port the old forms into
version-aware shape and kernel code; Phase E replaces this baseline-relative
classification with a full-history implementation manifest.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import onnx
except ImportError as exc:  # pragma: no cover - developer tool
    raise SystemExit("gen_opset_compat.py needs the development-only `onnx` package") from exc


ROOT = Path(__file__).resolve().parent.parent
CPU_TABLE = ROOT / "src" / "kernels" / "cpu_kernels.cpp"
OUTPUT = ROOT / "src" / "opset_compat_table.inc"
BASELINE_OPSET = 18
DOMAIN = ""

# New values for an existing attribute cannot be derived from the schema's
# attribute metadata (ONNX stores value domains in documentation). Each entry
# is reviewed and applied to every schema version at or after `since`.
VALUE_DOMAIN_OVERRIDES = {
    "Pad": (19, (("mode", "wrap"),), "Pad gained mode='wrap' at opset 19"),
}

# A changed default or accepted value set is a behavior change, not an
# additive attribute. Reject it until a kernel/canonicalization slice lands.
BEHAVIOR_OVERRIDES = {
    "GridSample": (
        20,
        "GridSample mode default and value domain changed at opset 20",
    ),
}


class GeneratorError(RuntimeError):
    pass


@dataclass(frozen=True)
class Entry:
    op_type: str
    version: int
    compatibility: str
    added_attributes: tuple[str, ...] = ()
    value_guards: tuple[tuple[str, str], ...] = ()
    reason: str = ""


def cpu_operators() -> list[str]:
    text = CPU_TABLE.read_text(encoding="utf-8")
    names = re.findall(r'\{\s*"([A-Za-z0-9_]+)"\s*,\s*&', text)
    unique = sorted(set(names))
    if not unique:
        raise GeneratorError(f"no operators found in {CPU_TABLE}")
    if len(unique) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise GeneratorError(f"duplicate CPU kernel rows: {duplicates}")
    return unique


def schema_history(op_type: str):
    history = [
        schema
        for schema in onnx.defs.get_all_schemas_with_history()
        if schema.domain == DOMAIN and schema.name == op_type
    ]
    if not history:
        raise GeneratorError(f"no ONNX schema for implemented operator {op_type!r}")
    return sorted(history, key=lambda schema: schema.since_version)


def input_signature(schema):
    return tuple((value.name, value.option.name) for value in schema.inputs)


def output_signature(schema):
    return tuple((value.name, value.option.name) for value in schema.outputs)


def attribute_signature(schema):
    return {
        name: (
            attribute.type.name,
            bool(attribute.required),
            attribute.default_value.SerializeToString(),
        )
        for name, attribute in schema.attributes.items()
    }


def added_attributes(baseline, schema) -> tuple[str, ...]:
    return tuple(sorted(set(schema.attributes) - set(baseline.attributes)))


def compare_schema(baseline, schema) -> None:
    """Fail on structural changes that have no reviewed override."""
    if input_signature(schema) != input_signature(baseline):
        raise GeneratorError(
            f"{schema.name} v{schema.since_version}: inputs changed relative to "
            f"v{baseline.since_version}; add a reviewed behavior override"
        )
    if output_signature(schema) != output_signature(baseline):
        raise GeneratorError(
            f"{schema.name} v{schema.since_version}: outputs changed relative to "
            f"v{baseline.since_version}; add a reviewed behavior override"
        )
    before = attribute_signature(baseline)
    after = attribute_signature(schema)
    removed = sorted(set(before) - set(after))
    changed = sorted(name for name in set(before) & set(after) if before[name] != after[name])
    if removed:
        raise GeneratorError(
            f"{schema.name} v{schema.since_version}: attributes removed: {removed}"
        )
    if changed:
        raise GeneratorError(
            f"{schema.name} v{schema.since_version}: attribute defaults/types changed: "
            f"{changed}; add a reviewed behavior override"
        )


def classify(op_type: str, baseline, schema) -> Entry:
    version = schema.since_version
    behavior = BEHAVIOR_OVERRIDES.get(op_type)
    if behavior is not None and version >= behavior[0]:
        return Entry(op_type, version, "BehaviorChange", reason=behavior[1])

    compare_schema(baseline, schema)
    added = added_attributes(baseline, schema)

    value_override = VALUE_DOMAIN_OVERRIDES.get(op_type)
    if value_override is not None and version >= value_override[0]:
        _, guards, reason = value_override
        return Entry(op_type, version, "ValueDomain", added, tuple(guards), reason)

    if added:
        return Entry(op_type, version, "AdditiveAttributes", added)
    return Entry(op_type, version, "Equivalent")


def build_entries() -> list[Entry]:
    entries: list[Entry] = []
    for op_type in cpu_operators():
        history = schema_history(op_type)
        baseline_candidates = [schema for schema in history if schema.since_version <= BASELINE_OPSET]
        if not baseline_candidates:
            raise GeneratorError(f"{op_type} has no schema at or before opset {BASELINE_OPSET}")
        baseline = baseline_candidates[-1]
        baseline_version = baseline.since_version
        for schema in history:
            version = schema.since_version
            if version < baseline_version:
                # Selectable by the resolver for older declarations, but not
                # reviewed as equivalent to the opset-18 implementation.
                entries.append(Entry(op_type, version, "Historical"))
            elif version == baseline_version:
                entries.append(Entry(op_type, version, "Equivalent"))
            else:
                entries.append(classify(op_type, baseline, schema))
    return entries


def cpp_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render(entries: list[Entry]) -> str:
    lines = [
        "// Generated by tools/gen_opset_compat.py. Do not edit by hand.",
        f"// ONNX schema source: onnx {onnx.__version__}.",
        "// Regenerate: python3 tools/gen_opset_compat.py",
        "",
    ]
    for entry in entries:
        added = ",".join(entry.added_attributes)
        guards = ",".join(f"{name}={value}" for name, value in entry.value_guards)
        lines.append(
            "NERT_OPSET_ENTRY("
            f"{cpp_string(entry.op_type)}, {entry.version}, {entry.compatibility}, "
            f"{cpp_string(added)}, {cpp_string(guards)}, {cpp_string(entry.reason)}"
            ")"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed table matches the generator instead of writing it",
    )
    args = parser.parse_args()

    entries = build_entries()
    generated = render(entries)

    if args.check:
        if not OUTPUT.exists():
            print(f"missing generated table {OUTPUT}", file=sys.stderr)
            return 1
        current = OUTPUT.read_text(encoding="utf-8")
        if current == generated:
            print(f"opset compatibility table is current ({len(entries)} entries)")
            return 0
        diff = difflib.unified_diff(
            current.splitlines(),
            generated.splitlines(),
            fromfile=str(OUTPUT),
            tofile="generated",
            lineterm="",
        )
        print("\n".join(diff), file=sys.stderr)
        return 1

    OUTPUT.write_text(generated, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
