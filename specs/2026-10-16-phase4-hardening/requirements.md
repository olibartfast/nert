# Requirements — Phase 4 Hardening + Policy

Date: 2026-10-16. Roadmap: Phase 4 (Planned). Constitution:
`specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md`.

## Goal

Close the three open policy questions and the Tier-5 close-out so Phase 5
(rename) and Phase N1 (CUDA) start from a decided baseline. No new ops.

## In Scope

- [R-1] [Q-1] opset policy: require opset-18 exports + a conversion note.
  Loader stays opset-18-only (no code change); README gains the exact
  `onnx.version_converter` command. Recommendation in tech-stack becomes
  the decision.
- [R-2] [Q-3] RAFT symbolic spatial dims: static per load ([D-8]) confirmed
  — Tier-4 RAFT parity already passed on it. Record as decided in
  tech-stack; no code change.
- [R-3] [Q-4] CI zoo subset: ResNet-18 fixture only (provisioned in CI);
  model-zoo parity stays in Neuriplo (`NertInferTest`,
  `nert_model_zoo_parity_test`). Record in tech-stack; no workflow change
  (CI already does exactly this).
- [R-4] Close-out: Tier-5 CHANGELOG entry (landed), roadmap Phase 3 status
  honest (kernels complete, parity gate open), validation evidence dated.
  Consumer note: Neuriplo adapter owns the parity runs, not this repo.

## Out of Scope

- New ops; namespace/filename rename (Phase 5); CUDA (N1); CI fixture
  changes; any `src/`/`test/` behavior change.

## Constraints

- Docs + spec edits only. `git diff --stat` must show no `src/`,
  `include/`, `test/`, `bench/` changes.
- SDD packet committed before any follow-up dispatch.

## Definition of Done

- [ ] [R-1]–[R-4] implemented, validation evidenced with dates
