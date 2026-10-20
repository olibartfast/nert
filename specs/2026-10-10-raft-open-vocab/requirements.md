# Requirements — RAFT + Open-Vocabulary Operator Coverage on NERT

Spec: `specs/2026-10-10-raft-open-vocab/requirements.md` · Roadmap: Phase 3
(Next) · Provenance: neuriplo `specs/2026-10-10-native-tasks-op-coverage/`
tiers 4–5 (tiers 1–3 landed in `olibartfast/nert@f9981db`).

## Goal

Close the two remaining operator tiers of the neuriplo-tasks gate inside the
standalone NERT repository: RAFT optical flow end to end, and OWLv2 +
Grounding DINO end to end — each loading on NERT and matching ONNX Runtime on
identical input through Neuriplo's model-zoo harness.

## In Scope

- [R-1] Tier 4 kernel: `InstanceNormalization` — opset-6 semantics:
  per-instance, per-channel normalization over spatial axes with `epsilon`,
  per-channel scale and bias. Loader allowlist entry + opset-18 attribute
  policy, shape rule, hand-computed unit tests incl. off-default `epsilon`
  and a negative case for a rejected configuration.
- [R-2] Tier 5 kernels (11): `ArgMax`, `Elu`, `LessOrEqual`, `ReduceL2`,
  `CumSum`, `EyeLike`, `GreaterOrEqual`, `Max`, `Min`, `Or`, `ScatterND` —
  each with shape inference, loader allowlist entry + opset-18 attribute
  policy, and hand-computed unit tests incl. off-default attributes and one
  negative load/shape case naming the op.
- [R-3] RAFT end to end: torchvision RAFT-large (2562 nodes, unrolled
  iterations, no `Loop` op; opset 16 source converted to 18 per [D-2]).
  Inputs `input1`/`input2` float32 `[N,3,H,W]` with symbolic H/W fixed per
  load ([D-8], [Q-3] = static per load); output `flow_prediction`. Passes
  `parity` (atol 1e-3 scaled) on a real image pair (bus.jpg 256×320 in
  [-1,1] + 4px-shifted copy) via Neuriplo's `nert_model_zoo_parity_test`.
- [R-4] Open-vocabulary end to end: OWLv2 and Grounding DINO (incl. `-tiny`
  checkpoint; same op set as `-base`) load and pass their manifest rows
  (parity/e2e per packet P-tier manifest), incl. int64 text inputs
  (`input_ids`, `attention_mask`) and int64 compute where the exports need
  it ([D-14] already landed; no new dtype design here).
- [R-5] Gaps closed generally at opset-18 semantics, not per-node
  workarounds. A gap needing a design change (new dtype, dynamic shapes,
  graph rewrite) stops and is reported — never worked around in the executor.
- [R-6] No existing test weakened; ResNet-18 fixture ctest stays green;
  YOLO + transformer + DETR regression manifests stay green.

## Out of Scope (with pointers)

- Performance; CPU kernels stay unoptimized oracles (Phase N1 owns speed).
- Gaussian splatting (LGM, GRM) and VLM (llama.cpp) — outside the gate.
- [Q-1] opset policy, [Q-4] CI zoo subset, [Q-5] namespace rename — Phase 4/5,
  not this packet.
- CUDA kernels — Phase N1.
- Neuriplo-side changes (adapter, manifests, harness): this repo ships the
  engine; parity evidence is recorded from Neuriplo runs, and any harness gap
  (e.g. multi-file `;` input column from P4a) is reported back, not
  implemented here.

## Constraints (traced to constitution)

- `specs/mission.md`: interpreter shape, one-device-per-graph, oracle-not-target.
- `specs/tech-stack.md`: stdlib-only runtime; opset-18 loader; float32
  compute with int64/bool data+I/O; double accumulation [D-15]; shape seam
  (per-shape planning, no parser/IR/kernel change); [D-8] symbolic-dims rule.
- Orchestration discipline holds (see `orchestration.md`): packet committed
  before dispatch, one ledger row per attempt, worker never commits,
  orchestrator re-scores.

## Open Questions (inherited)

- [Q-1] Opset 12–17 native vs require-18 + conversion note (Phase 4).
- [Q-3] RAFT spatial dims: static per load (recommended) vs plan cache.
  This packet proceeds on static-per-load; a failure here reopens the question
  instead of smuggling in dynamic machinery.
- [Q-4] Which zoo subset, if any, runs in CI (Phase 4).

## Definition of Done (requirements level)

- [ ] Every [R-n] implemented or explicitly deferred with a tracked location
- [ ] No out-of-scope work smuggled in; no in-scope behavior silently dropped
