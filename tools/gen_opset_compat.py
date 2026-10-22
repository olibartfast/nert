#!/usr/bin/env python3
"""Generate the committed full-history ONNX schema catalog.

Development-only tool. The runtime never imports ONNX; it compiles the
generated `src/opset_compat_table.inc`.

Usage:
    python3 tools/gen_opset_compat.py          # rewrite the table
    python3 tools/gen_opset_compat.py --check  # fail if the table is stale

The table covers every operator in `src/kernels/cpu_kernels.cpp`. For each
operator it records every schema version in ONNX's history. The reviewed
implementation manifest in `tools/opset_manifest.py` names the attributes the
runtime actually implements; every other attribute in a schema becomes a
load-time guard. Whole-schema behavior changes and reviewed value-domain guards
are also declared in the manifest. There is no baseline opset: each schema is
classified against the implementation, not against a canonical version.

Input/output signature changes and attribute type/default/required changes must
be reviewed in the manifest. An ONNX update that changes the surface without a
review fails generation instead of silently widening what the runtime accepts.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import onnx
except ImportError as exc:  # pragma: no cover - developer tool
    raise SystemExit("gen_opset_compat.py needs the development-only `onnx` package") from exc

ROOT = Path(__file__).resolve().parent.parent
CPU_TABLE = ROOT / "src" / "kernels" / "cpu_kernels.cpp"
OUTPUT = ROOT / "src" / "opset_compat_table.inc"
DOMAIN = ""

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from opset_manifest import (  # noqa: E402
    ATTRIBUTE_CHANGE_REVIEWS,
    BEHAVIOR_CHANGES,
    FORM_REVIEWS,
    IMPLEMENTED_ATTRIBUTES,
    VALUE_DOMAIN_GUARDS,
)


class GeneratorError(RuntimeError):
    pass


@dataclass(frozen=True)
class Entry:
    op_type: str
    version: int
    compatibility: str
    unsupported_attributes: tuple[str, ...] = ()
    guarded_values: tuple[tuple[str, str], ...] = ()
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


def attribute_changes(previous, schema) -> tuple[str, ...]:
    before = attribute_signature(previous)
    after = attribute_signature(schema)
    return tuple(
        sorted(
            name
            for name in set(before) & set(after)
            if before[name] != after[name]
        )
    )


def check_history_reviews(op_type: str, previous, schema) -> None:
    version = schema.since_version
    if (
        input_signature(previous) != input_signature(schema)
        or output_signature(previous) != output_signature(schema)
    ) and (op_type, version) not in FORM_REVIEWS:
        raise GeneratorError(
            f"{op_type} v{version}: input/output signature changed relative to "
            f"v{previous.since_version}; add a FORM_REVIEWS entry"
        )
    changed = attribute_changes(previous, schema)
    if changed and (op_type, version) not in ATTRIBUTE_CHANGE_REVIEWS:
        raise GeneratorError(
            f"{op_type} v{version}: attribute type/default/required changed "
            f"({', '.join(changed)}); add an ATTRIBUTE_CHANGE_REVIEWS entry"
        )


def behavior_reason(op_type: str, version: int) -> Optional[str]:
    for first, last, reason in BEHAVIOR_CHANGES.get(op_type, ()):
        if version >= first and (last is None or version <= last):
            return reason
    return None


def value_guard(op_type: str, version: int):
    for first, last, guards, reason in VALUE_DOMAIN_GUARDS.get(op_type, ()):
        if version >= first and (last is None or version <= last):
            return guards, reason
    return None


def validate_manifest(cpu_ops: list[str]) -> None:
    missing = sorted(set(cpu_ops) - set(IMPLEMENTED_ATTRIBUTES))
    extra = sorted(set(IMPLEMENTED_ATTRIBUTES) - set(cpu_ops))
    if missing:
        raise GeneratorError(f"implementation manifest is missing operators: {missing}")
    if extra:
        raise GeneratorError(f"implementation manifest names non-CPU operators: {extra}")

    for op_type in cpu_ops:
        attributes = IMPLEMENTED_ATTRIBUTES[op_type]
        if len(attributes) != len(set(attributes)):
            raise GeneratorError(f"{op_type}: duplicate implemented attributes in manifest")
        schema_attributes = set()
        for schema in schema_history(op_type):
            schema_attributes.update(schema.attributes)
        unknown = sorted(set(attributes) - schema_attributes)
        if unknown:
            raise GeneratorError(
                f"{op_type}: manifest names attributes absent from every schema: {unknown}"
            )


def build_entries() -> list[Entry]:
    cpu_ops = cpu_operators()
    validate_manifest(cpu_ops)

    entries: list[Entry] = []
    for op_type in cpu_ops:
        implemented = set(IMPLEMENTED_ATTRIBUTES[op_type])
        history = schema_history(op_type)
        previous = None
        for schema in history:
            version = schema.since_version
            if previous is not None:
                check_history_reviews(op_type, previous, schema)

            behavior = behavior_reason(op_type, version)
            if behavior is not None:
                entries.append(Entry(op_type, version, "BehaviorChange", reason=behavior))
                previous = schema
                continue

            unsupported = tuple(sorted(set(schema.attributes) - implemented))
            value = value_guard(op_type, version)
            if value is not None:
                guards, reason = value
                entries.append(
                    Entry(op_type, version, "ValueDomain", unsupported, tuple(guards), reason)
                )
            elif unsupported:
                entries.append(Entry(op_type, version, "GuardedAttributes", unsupported))
            else:
                entries.append(Entry(op_type, version, "Implemented"))
            previous = schema
    return entries


def cpp_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render(entries: list[Entry]) -> str:
    lines = [
        "// Generated by tools/gen_opset_compat.py. Do not edit by hand.",
        f"// ONNX schema source: onnx {onnx.__version__}.",
        "// Implementation manifest: tools/opset_manifest.py.",
        "// Regenerate: python3 tools/gen_opset_compat.py",
        "",
    ]
    for entry in entries:
        unsupported = ",".join(entry.unsupported_attributes)
        guards = ",".join(f"{name}={value}" for name, value in entry.guarded_values)
        lines.append(
            "NERT_OPSET_ENTRY("
            f"{cpp_string(entry.op_type)}, {entry.version}, {entry.compatibility}, "
            f"{cpp_string(unsupported)}, {cpp_string(guards)}, {cpp_string(entry.reason)}"
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
            print(f"opset schema catalog is current ({len(entries)} entries)")
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
