# Requirements — Newer-than-18 opsets via verified compatibility

Date: 2026-10-21. Sibling and dependency:
`specs/2026-10-21-opset-range/` (declared opsets ≤18, shared opset plumbing).
This packet owns declarations above 18. It borrows OpenCV 5's acceptance
model — the declared opset is data, not a hard gate — while keeping NERT's
fail-fast correctness rule.

## Problem

The loader rejects any declared `ai.onnx` opset other than 18. Newer exports
declare 19 and above (the ONNX reference version is 27 as of the 2026-10-21
audit), even when every operator in the graph has the same semantics as the
opset-18 form NERT implements. Requiring a down-conversion is avoidable work
and conversion is not proof of equivalence.

OpenCV 5 accepts any declared opset. Its ONNX importer records the version
per domain (`ONNXImporter2::parseOperatorSet`), builds one dispatch map, and
branches only where semantics changed (`Softmax` below 13, `Resize` before
16). Old and new syntactic forms are absorbed by the layers. There is no
maximum-opset check, and unknown attributes are generally passed through and
ignored by the layers.

That breadth is useful, but the silent-ignore part conflicts with NERT's
mission: unsupported paths must fail loudly, never be approximated. This
packet therefore copies OpenCV's *acceptance breadth* and replaces its
silent fallback with a verified compatibility table plus explicit guards.

## Goal

Accept a model whose declared opset is above 18 when every node resolves to
a schema version NERT has verified as executable by the opset-18 kernels. A
node that uses a newer attribute, a newer value domain, a new operator, or a
new dtype is rejected at load with a message naming the node, operator,
effective version, and the missing capability.

## In Scope

- [R-1] **Declared opset above 18 is accepted as data.** There is no fixed
  numeric ceiling in the loader. The per-node compatibility check is the
  contract.
- [R-2] **Effective schema version.** For each node, resolve
  `effective = max { version in schema_versions[op] : version <= declared }`
  from a committed table generated from ONNX's schema history. The declared
  version alone is not used to select behavior.
- [R-3] **Compatibility classes.** Each schema version relative to the
  opset-18 implementation is classified as one of:
  - `equivalent` — attributes, inputs, and defaults are identical; accept.
  - `additive-attributes` — only new attributes were added; accept only
    when none of them is present (the allowlist rejects them otherwise).
  - `value-domain` — the same attributes accept new values; accept only
    when the node does not use a new value; otherwise reject.
  - `behavior-change` — a default or computation changed; reject until a
    kernel or canonicalization rule implements it.
  - `new-operator` — not implemented; reject as an unknown operator.
- [R-4] **Explicit guards for known post-18 changes.** At minimum:
  - `Cast` at 19+: reject `saturate`; at 24+: reject `round_mode`. A plain
    `Cast` with only `to` remains accepted because its behavior is unchanged.
  - `Pad` at 19+: reject `mode="wrap"`; accept `constant`, `reflect`, and
    `edge`.
  - Any further value-domain change discovered by the table generator gets a
    guard in the same change; no new value reaches a kernel by accident.
- [R-5] **Unsupported dtypes and operators stay rejected.** New element
  types (for example float8/int4) and operators introduced after the
  implemented set fail at load with the existing explicit messages.
- [R-6] **Fail-fast diagnostics.** Every rejection names the node, operator,
  declared opset, effective version, and the reason. No unknown attribute is
  ignored and no unsupported value is approximated.
- [R-7] **Canonical execution.** The kernels continue to implement the
  opset-18 semantics; no per-kernel or per-device version branches are
  added. Nodes accepted under this packet are exactly those whose resolved
  schema is executable by that implementation.
- [R-8] **Tests and gates.** Synthetic models at declared 19, 21, and 27;
  positive cases using only stable operators; negative cases for
  `Cast.saturate`, `Cast.round_mode`, `Pad.wrap`, a new operator, and an
  unsupported dtype; one real newer-than-18 model compared with ONNX
  Runtime.
- [R-9] **Docs and roadmap.** `README.md`, `specs/tech-stack.md` (`Q-1`),
  `specs/2026-10-17-task-models/coverage.md`, `CHANGELOG.md`, and
  `specs/roadmap.md` describe the verified-compatibility policy.

## Out of Scope

- Implementing new operators or new dtypes because a newer model uses them.
  Each gets a normal operator packet with kernel, shape, tests, and parity.
- Implementing `Cast.saturate`, `Cast.round_mode`, or `Pad.wrap`. This
  packet rejects them explicitly; implementing one is a follow-up slice of
  this packet with its own evidence.
- Silent attribute ignoring, per-operator fallback, or adopting OpenCV's
  “try anything” behavior.
- Non-`ai.onnx` domains and custom operators; their handling is unchanged.
- Declared opsets below the floor owned by
  `specs/2026-10-21-opset-range/`.
- Dynamic shapes, quantization, or new device backends.

## Decisions

- [D-1] **Breadth with proof, not breadth by hope.** Adopt OpenCV's
  premise that a declaration is metadata; verify each node against a table
  and guard instead of letting handlers ignore what they do not understand.
- [D-2] **One generated table, committed and reviewed.** A development-time
  generator reads ONNX schema history and emits a compact per-operator
  version/compatibility table. The runtime stays standard-library-only; the
  generator is never linked.
- [D-3] **Effective version, not declared version.** A declared-27 model
  using `Add` resolves to schema 14 and is accepted; a declared-19 model
  using `Cast` resolves to 19 and is checked against the Cast guards.
- [D-4] **Keep the strict attribute allowlist.** New attributes are not
  admitted just because a newer opset allows them; `saturate`, `round_mode`,
  and any future attribute need an explicit implementation slice.
- [D-5] **No arbitrary numeric ceiling.** The table is the contract. When a
  future ONNX release adds a schema, rerunning the generator and reviewing
  the diff is the update path.
- [D-6] **Depend on the range packet.** Shared parsing of `opset_import`
  and the canonicalization seam land in
  `specs/2026-10-21-opset-range/` Phase A. This packet starts after that
  phase is merged; it does not duplicate the parser.
- [D-7] **OpenCV's `ENGINE_ORT` escape hatch is not copied.** NERT has no
  second engine; a node either passes the compatibility check or fails.

## Constraints

- C++17, standard-library-only runtime; no new link dependency.
- One device per graph; no executor, planner, or kernel-table changes.
- Existing opset-18 behavior and all current model gates remain unchanged.
- Load-time rejection with a useful message is the only failure mode for
  unsupported newer features.
- The schema table must be reproducible and reviewable: a regenerated table
  with no semantic change yields an empty diff.

## Dependencies

- `specs/2026-10-21-opset-range/` Phase A: declared-opset parsing, range
  check, canonicalization hook.
- `src/model_loader.cpp`: per-operator attribute allowlist and load-time
  rejection points.
- Development-only `onnx` (1.22.0 at audit time) for the table generator.
- The model-zoo harness for the real newer-than-18 gate.

## Context

- OpenCV 5, current `5.x`: `ONNXImporter2::parseOperatorSet` stores the
  maximum declared version per domain; `buildDispatchMap_ONNX_AI` is a
  single op-keyed map; `parseSoftMax` branches for opset `<13`;
  `parseResize2` detects opset-10/11/13 input forms and defaults
  `coordinate_transformation_mode` from the opset; `Cast2` reads `to` but
  not the opset-19 `saturate` attribute. This is the behavior the packet
  deliberately improves on.
- Audit of the 72 implemented operators (onnx 1.22.0, 2026-10-21): after
  opset 18 the only attribute/input/default changes are `Cast` (`saturate`
  at 19, `round_mode` at 24) and the `Pad` `mode` value `wrap` at 19.
  `Resize` 19 and `ReduceMax` 20 entries are type/documentation changes;
  `ScatterND` and `Split` have no later schema. That makes the first
  compatibility table small.
- The model zoo currently pins opset 18, so this packet adds compatibility
  without changing a shipped gate.

## Assumptions & Open Questions

- [A-1] The post-18 audit above is complete for the implemented operator
  set; the generator will re-derive it and any mismatch blocks landing.
- [A-2] Type-constraint additions after 18 do not matter while the loader
  rejects dtypes outside float32/int64/bool/int32-emulation.
- [Q-1] Should a plain `Cast` in a declared-27 model be accepted (the node
  does not use `saturate`/`round_mode`)? Recommendation: yes, with explicit
  attribute guards, because the computation is unchanged. The alternative
  is to mark all `Cast` versions above 18 unverified, which is simpler but
  rejects safe models.
- [Q-2] Should `Pad.mode="wrap"` be implemented now? Recommendation: no;
  guard it and add the mode in a separate slice with tests and a model.
- [Q-3] Should the table include operators NERT does not implement, so a
  newer model fails with “operator not implemented” rather than “schema
  version unverified”? Recommendation: no; the existing unknown-operator
  path is already explicit.
- [Q-4] At integration, should the real gate use a model exported natively
  at 19+ or a converted one? Recommendation: a native export; conversion
  would not prove the newer-declaration path.

## Definition of Done (requirements level)

- [ ] Every [R-n] is implemented or explicitly deferred with a tracked
      location.
- [ ] No newer attribute, value, dtype, or operator is silently accepted.
- [ ] The compatibility table is generated, committed, reviewed, and
      reproducible.
