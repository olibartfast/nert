# Requirements — Opset range (≤18) via load-time canonicalization

Date: 2026-10-21. Follow-up to Phase 4 decision `Q-1` (exact opset 18).
Sibling packet: `specs/2026-10-21-opset-newer-than-18/` owns declarations
above 18. This packet owns the shared opset plumbing and the ≤18 path.

## Problem

The loader requires the declared `ai.onnx` opset to be exactly 18
(`src/model_loader.cpp:110`, check at `:1760`). A model exported at opset
13–17 is rejected even when every operator it uses has the same semantics
as the opset-18 form NERT implements. The current answer is to convert the
model with `onnx.version_converter`; that is extra tooling and conversion is
not proof of equivalence.

A schema audit of the 72 implemented operators (onnx 1.22.0 definitions,
2026-10-21) found that between opsets 13 and 18 the only non-additive change
is:

- `ReduceMean`, `ReduceL2`, `ReduceMax`: `axes` is an **attribute** at
  opsets 13–17 and becomes an **input** at 18.

Every other 13→18 change is an added attribute (which an older model cannot
use), an added optional input (same), or a new operator (which an older
model cannot contain). Below 13 the syntactic breaks multiply: `Softmax`
changes coercion and default axis at 13, and `ReduceSum` / `Squeeze` /
`Unsqueeze` / `Split` move attributes to inputs at 13.

## Goal

Accept declared `ai.onnx` opsets 13–18 natively (Phase A) and 11–12
(Phase B) by rewriting old node forms into the existing opset-18 IR at load
time. Shape inference, planning, the executor, and every device kernel keep
seeing exactly one canonical form. Models outside the supported range are
rejected with a message naming the node, operator, declared opset, and the
supported range.

## In Scope

- [R-1] **Range policy.** Parse the declared `ai.onnx` opset once. Phase A
  accepts 13–18; Phase B extends the floor to 11. Anything below the active
  floor and anything above 18 is rejected at load.
- [R-2] **Canonicalization stage.** A single load-time pass runs after node
  and attribute decoding and before constant folding and shape inference.
  It receives the declared opset and rewrites only the node forms listed
  below; node order and tensor names that are not part of a rewrite stay
  unchanged. The output is the current opset-18 IR.
- [R-3] **Phase A rules (13–17).**
  - `ReduceMean` / `ReduceL2` / `ReduceMax` with an `axes` attribute:
    create a unique int64 initializer for the axes, append it as the second
    node input, and drop the attribute.
  - `ScatterND` with a `reduction` attribute (added at 16): accept
    `"none"` by dropping the attribute; reject every other value, because
    the kernel implements only the overwrite form.
- [R-4] **Attribute admission.** The loader's per-operator allowlist admits
  `axes` for those three reductions and `reduction` for `ScatterND` so the
  normalizer can inspect them. The normalizer rejects the old forms when the
  declared opset makes them illegal (for example an `axes` attribute on an
  opset-18 `ReduceMean`).
- [R-5] **Phase B rules (11–12).**
  - `Softmax` below 13: preserve the old semantics by expanding to
    `Flatten(axis)` → `Softmax(axis=1)` → `Reshape(original shape)`, and use
    the old default `axis = 1` when the attribute is absent.
  - `ReduceSum` with an `axes` attribute: convert to an int64 input.
  - `Squeeze` / `Unsqueeze` with an `axes` attribute: convert to an int64
    input.
  - `Split` with a `split` attribute: convert to an int64 input.
  - `Resize` 11-form input layout: canonicalize only if discovery shows it
    differs from the 18 form; otherwise no rule is added.
- [R-6] **Fail-fast diagnostics.** Every rejection names the node, the
  operator, the declared opset, the supported floor/ceiling, and the reason.
  No unsupported attribute or old form is silently ignored.
- [R-7] **Deterministic graph.** Injected initializer names are unique and
  stable (`<node>/axes`, with a numeric suffix on collision), the tensor
  table records them, and no unrelated tensor or node is renamed.
- [R-8] **Policy docs.** `README.md`, `specs/tech-stack.md` (`Q-1`),
  `specs/2026-10-17-task-models/coverage.md`, `specs/roadmap.md`, and
  `CHANGELOG.md` state the new policy and its phases. Offline conversion
  remains the recommendation below the supported floor.

## Out of Scope

- Declared opsets above 18: `specs/2026-10-21-opset-newer-than-18/`.
- Opsets 10 and below. They need `Slice`/`Clip`/`Pad`/`TopK` attribute forms,
  `Upsample`, and older `Gemm` semantics. Deferred with a roadmap entry; no
  implementation here.
- New operators, new dtypes, `Split.num_outputs`, and the opset-18
  `Resize` attributes (`axes`, `keep_aspect_ratio_policy`) not already
  admitted. They are existing coverage gaps, not opset-range work.
- Per-kernel or per-device version branches. Kernels keep one form.
- Runtime dependency on `onnx` or protobuf. Any schema-table generator is
  development tooling and its output is committed.
- Silent attribute ignoring, per-operator fallback, or dynamic shapes.

## Decisions

- [D-1] **Canonicalize to the opset-18 IR at load.** One implementation per
  operator across CPU, CUDA, OpenCL, and HIP. Per-kernel version branches
  would multiply the work by four and make parity harder to audit.
- [D-2] **Two thin phases:** A = 13–18, B = 11–12. A needs one rewrite rule;
  B needs five. Land and gate A before touching B.
- [D-3] **Keep the strict attribute allowlist.** Admission is widened only
  for attributes the normalizer consumes, and the normalizer rejects them in
  the versions where they are illegal.
- [D-4] **No public IR change.** The declared opset and the normalization
  stay inside the loader; `Graph`, `Node`, and `TensorView` keep their
  current layout.
- [D-5] **Conversion remains the fallback.** Models below the supported
  floor are still converted offline and gated end-to-end; NERT does not
  auto-convert arbitrary old graphs.
- [D-6] **Correctness is proven by model gates, not by the schema audit.**
  The audit only scopes the work; every accepted version family gets a real
  model comparison against ONNX Runtime.

## Constraints

- Runtime stays C++17 and standard-library-only; no new link dependency.
- One device per graph; no executor, planner, or shape-inference policy
  changes.
- Existing opset-18 behavior and numerics are unchanged: all current model
  gates and test suites must stay green.
- Static shapes, load-time failure, and useful error messages remain part of
  the contract.

## Dependencies

- `src/model_loader.cpp`: opset decode, per-operator attribute allowlist,
  decode order relative to constant folding.
- `include/nert/graph.hpp`: `Node` and `Initializer` (no layout change
  planned).
- `src/shape_inference.cpp` and all kernel tables: consumers of the
  canonical form; expected to need no change.
- `test/loader_test.cpp`, `test/shape_inference_test.cpp`,
  `test/kernels_test.cpp`, `test/CMakeLists.txt`: synthetic wire-encoded
  models for each accepted/rejected version.
- Development-only `onnx` (1.22.0) for schema auditing.

## Context

- The exact-18 policy is recorded in `specs/tech-stack.md` `Q-1` and
  `README.md` under “Opset policy”.
- The loader is a hand-written protobuf reader; it already extracts
  `opset_import` and compares the version. The change is a range check plus
  a normalization pass, not a new parser.
- OpenCV 5 solves the same problem with a per-domain opset map and dual-form
  parsers. This packet adopts the dual-form *outcome* at load time while
  keeping NERT's fail-fast attribute checking; the OpenCV comparison and the
  newer-than-18 path are in the sibling packet.
- The model zoo is pinned at opset 18 today, so the new range adds
  compatibility without changing any existing gate.

## Assumptions & Open Questions

- [A-1] For 13–17, the reductions above are the only rewrites required.
  Basis: the 2026-10-21 schema audit; validate with real exports in Phase A.
- [A-2] Dropping `ScatterND.reduction = "none"` is semantically exact; the
  other reduction values are not implemented.
- [Q-1] Should Phase B include opset 10? Recommendation: stop at 11, because
  10 still needs `Clip`/`Pad` attribute forms and the 10-form `Resize`.
- [Q-2] For `Softmax` below 13, is the Flatten/Reshape expansion accepted,
  or should those models be rejected with a conversion message?
  Recommendation: expand, because the old semantics are well defined and
  the expansion is testable.
- [Q-3] Normalizer placement: a new `src/opset_normalize.cpp` with an
  internal header, called from `decode_model`, versus adding it to
  `model_loader.cpp`. Recommendation: separate file, so the loader keeps
  one responsibility and tests can drive the pass directly.

## Definition of Done (requirements level)

- [ ] Every [R-n] is implemented or explicitly deferred with a tracked
      location.
- [ ] No in-scope behavior is silently dropped; no out-of-scope work is
      smuggled in.
- [ ] The supported range is stated in code, tests, README, and the
      constitution.
