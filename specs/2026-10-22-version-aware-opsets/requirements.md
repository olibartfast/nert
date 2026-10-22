# Requirements — Version-aware opsets

Date: 2026-10-22. Supersedes the canonical-opset-18 decisions `D-1` and `D-4`
in `specs/2026-10-21-opset-range/`.

## Problem

The runtime currently has one global semantic target: opset 18. Declarations
11–18 are rewritten into the “opset-18 IR” by
`src/opset_normalize.cpp`; declarations above 18 are accepted only when every
node is classified as equivalent to that same implementation. This makes `18`
a policy constant shared by the loader, the generator, the shape code, the
kernels, and the tests. It also blocks newer semantics that cannot be lowered
to exactly the opset-18 form.

ONNX versions are per-operator schema facts, not a single semantic switch.
The runtime should carry the effective schema on each node and let each layer
resolve the behavior it needs. This is the same direction as OpenCV 5's new
DNN engine: a typed operation graph with shape inference and explicit
operator semantics, not a global rewrite of every model to one opset.

## Goal

Replace global canonicalization with a version-aware graph:

- `Graph.declared_opset` records the model declaration.
- `Node.schema_version` records the effective ONNX schema resolved from the
  declared opset and the committed schema catalog.
- Shape inference and kernels resolve attribute/input forms and semantic
  modes from `Node.schema_version` through shared helpers.
- The generated catalog covers the full schema history used by the accepted
  declaration range, not only versions at or above 18.
- Unsupported capabilities still fail loudly with node, operator, declared
  opset, effective schema, and the missing capability.

## In Scope

- [R-1] **Per-node schema identity.** Every node decoded from the model
  carries the effective schema version selected by
  `effective = max { version in schema_versions[op] : version <= declared }`.
  The graph records the declared `ai.onnx` opset. `Constant` nodes are folded
  into initializers before a `Node` exists and are therefore exempt. Nodes
  synthesized by a remaining load-time rewrite may carry `0` until that
  rewrite is removed in Phase B–D; they are not source nodes. Because a
  remaining rewrite may change the node form after this field is stamped,
  the field is the source schema until that operator's rewrite is removed.
- [R-2] **Full-history catalog.** The generator emits every schema version
  needed to resolve the accepted declaration range (11 and above) for every
  implemented operator, not only versions at or above 18. The runtime table
  remains internal and committed; generation remains development-only.
- [R-3] **No global canonical opset.** Remove `NormalizeGraphForOpset`, the
  `opset-18 IR` policy, and `BASELINE_OPSET = 18` from the generator. Any
  remaining version constants name a specific ONNX schema change and live in
  one semantic-resolution place.
- [R-4] **Version-aware form resolution.** Reduce axes, `Squeeze` /
  `Unsqueeze` axes, `Split` split, and `ScatterND.reduction` are read from
  the form legal for the node's effective schema. No graph rewrite is needed
  to make shape inference or a kernel understand an old form.
- [R-5] **Version-aware behavior.** Legacy `Softmax` below schema 13 is
  implemented as a 2-D coercion over the trailing axes, directly in the
  device kernels. It does not require a statically known rank and does not
  expand to `Shape → Flatten → Softmax → Reshape`.
- [R-6] **Strict admission.** A node is accepted only when its effective
  schema is implemented. Unknown attributes, illegal forms, unsupported
  values, and unimplemented schema behavior remain explicit load failures.
  No OpenCV-style automatic fallback or ONNX Runtime escape hatch is added.
  Phase A resolves schemas and rejects operators introduced after the
  declaration; full attribute/form admission keyed by schema is completed
  in Phases B–C. The pre-existing allowlist holes (for example, an attribute
  introduced after the node's effective schema) are not claimed fixed by
  Phase A.
- [R-7] **One semantic implementation per mode.** CPU, CUDA, OpenCL, and HIP
  share the same named semantic modes and resolution helpers. A kernel may
  branch once at entry; it must not branch on a version inside an element
  loop.
- [R-8] **Behavior preservation.** Existing opset-18 results, model gates,
  diagnostics, and public entry points remain unchanged except for the
  deliberate `Node` / `Graph` metadata addition and the new rejection of a
  source node whose operator has no schema at or below the declared opset
  (an ONNX-invalid model that the previous loader accepted).

## Out of Scope

- New operators, new dtypes, dynamic shapes, control flow, and QDQ graphs.
- OpenCV's `ENGINE_AUTO`, `ENGINE_CLASSIC`, or `ENGINE_ORT` fallback model.
- Implementing the post-18 capability slices (`Cast.saturate`,
  `Cast.round_mode`, `Pad.wrap`, `AveragePool.dilations`, `Range.stash_type`,
  `GridSample` 20+). They remain explicitly rejected until their own slices.
- Declared opsets below 11.
- A public plugin or capability registry.

## Decisions

- [D-1] **Version-aware nodes replace canonical opset 18.** This reverses
  `D-1` and `D-4` of the opset-range packet.
- [D-2] **Keep the public `Node` / `Graph` change small.** Add an effective
  schema value to `Node` and the declared opset to `Graph`; do not add an
  operator-semantics variant to the IR.
- [D-3] **Named semantic modes, not raw version checks in kernels.** Closed
  behavior differences use `enum class` plus a resolver. Version comparisons
  live in the resolver, not in loops or per-device tables.
- [D-4] **The catalog is an implementation gate, not a compatibility verdict
  relative to 18.** It resolves schemas and reports whether a capability is
  implemented.
- [D-5] **Migrate by vertical capability.** Land metadata and the catalog
  first; then remove rewrites one operator family at a time with shadowed
  old/new output comparisons.

## Constraints

- C++17, standard library only. No new runtime dependency.
- The ONNX package and generator remain development-only.
- `Node` is public; the layout change is accepted for the unreleased 0.2.0
  and must be documented.
- One device per graph; no executor or planner policy change.
- Validation is model- and device-backed; a schema audit alone is not proof.

## Acceptance

Every phase must keep the existing opset-18 model gate green and add the
targeted version-aware checks for the forms it moves. The final state has no
`opset_normalize.cpp`, no `NormalizeGraphForOpset` call, and no global
`BASELINE_OPSET`.
