# Plan — Opset range (≤18) via load-time canonicalization

One worktree per phase (`wip/nert-opset-range-a`, `...-b`). Phase A lands
first and proves the range plumbing and the canonicalization stage before
Phase B adds the older syntax rules. Every task group ends with a runnable
check; the model gate is the integration checkpoint.

## Group 0 — Index and constitution

- [T-1] Add this packet and the sibling newer-than-18 packet to
  `specs/roadmap.md`, with status `planned` and the dependency between them.
  Deliverable: roadmap names both specs.
- [T-2] Add a “revision in progress” note to `specs/tech-stack.md` `Q-1`
  pointing at both packets. The exact-18 gate stays in force until Phase A
  is validated. Deliverable: constitution and code agree on the current
  state.
- [T-3] Add the input-format note to `README.md` only when Phase A lands;
  do not advertise unshipped support.

## Group 1 — Range plumbing (Phase A)

- [T-4] Change `decode_model` to retain the declared `ai.onnx` version
  instead of reducing it to a boolean. Keep the existing rejection when the
  import is missing or malformed.
  - Checks: `loader_test` still passes; a missing-import model still fails.
- [T-5] Replace the equality check with a range check for 13–18. The error
  message names the declared version, the node (when known), and
  `supported opsets 13..18`.
  - Checks: synthetic opset-13 and opset-18 models load; 12 and 19 fail
    with the new message.
- [T-6] Add the canonicalization call after the graph body decode and
  folded-constant materialization, before shape inference. Pass the declared
  opset and the parsed graph. No-op for 18.
  - Checks: an opset-18 graph is byte-for-byte unchanged in node order,
    inputs, attributes, and initializers.
- [T-7] Add tests for [R-1], [R-6], [R-7] in `test/loader_test.cpp`
  (positive 13/18, negative 12/19, missing opset, malformed version).

## Group 2 — Canonicalization (Phase A)

- [T-8] Create `src/opset_normalize.cpp` (internal header beside it) with
  `NormalizeGraphForOpset(Graph&, int64_t declared_opset)`. Add it to the
  library sources and the test target. The file owns only version rewrites;
  it does not re-parse ONNX.
- [T-9] Implement [R-3] for `ReduceMean`/`ReduceL2`/`ReduceMax`:
  - locate `axes` in `node.attributes`;
  - validate it is an int list with in-range axes;
  - create an int64 initializer with a collision-free name and a tensor
    table entry;
  - append the input and erase the attribute;
  - reject a second input or an illegal attribute at opset 18.
- [T-10] Implement [R-3] for `ScatterND`: admit `reduction`, drop `"none"`,
  reject other values with the node name.
- [T-11] Implement [R-4] allowlist changes and the version checks that make
  an old attribute illegal at 18.
- [T-12] Unit tests:
  - an opset-13 `ReduceMean` with `axes=[1]` becomes a two-input node and
    produces the same values as the opset-18 input form;
  - the same graph at opset 18 with an `axes` attribute is rejected;
  - `ScatterND` with `reduction="none"` is accepted and with `"add"` is
    rejected;
  - collision naming, empty axes, duplicate axes, out-of-range axes.
  - Checks: focused `nert_loader`, `nert_shapes`, `nert_kernels` tests.

## Group 3 — Phase A integration

- [T-13] Provision one real opset-13 model (ResNet-18 export or a small
  deterministic graph with a ReduceMean) and one opset-17 model. Run both
  through NERT and compare against an ONNX Runtime reference for the same
  weights and inputs.
  - Checks: outputs within the established `1e-3·(1+|ref|)` bar; record
    hashes and commands in `validation.md`.
- [T-14] Run the full existing suite: OFF, CUDA with cuDNN, OpenCL, HIP
  where available, plus `scripts/format.sh --check`.
- [T-15] Update `README.md` “Opset policy”, `CHANGELOG.md`, and the roadmap
  status. `Q-1` moves from exact-18 to 13–18 after Phase A, then to 11–18
  after Phase B.

## Group 4 — Phase B (11–12)

- [T-16] Implement the `Softmax` rewrite from [R-5]: `Shape` for the
  original dims, Flatten with the default/axis value, Softmax on axis 1,
  Reshape back. `Shape` avoids needing a static shape constant for an
  intermediate input; the input rank must be known from `value_info`.
- [T-17] Implement the other Phase B rewrites: `ReduceSum` axes attribute,
  `Squeeze`/`Unsqueeze` axes attribute, `Split` split attribute. Extend the
  allowlist only for the consumed attributes.
- [T-18] Add positive/negative tests per operator and one real opset-11 or
  opset-12 model gate against ONNX Runtime. Confirm that opset 10 still
  fails with the range message and the conversion pointer.

## Group 5 — Integration and handoff

- [T-19] Update the model-coverage note
  (`specs/2026-10-17-task-models/coverage.md`) and the tech-stack `Q-1`
  wording to the final supported range.
- [T-20] Re-run every automated check in `validation.md`; record evidence
  and any deviations.
- [T-21] Leave the >18 packet unblocked: the declared-opset parsing and the
  canonicalizer seam are the shared dependency it needs.

## Notes

- Reuse existing loader helpers for attributes and initializers; do not
  introduce a second protobuf path.
- The normalizer must be deterministic and idempotent for opset 18.
- If discovery shows another 13–17 form difference (a real model, not a
  schema guess), stop and replan this packet before adding a rule by
  assumption.
