# Plan — RAFT + Open-Vocabulary Operator Coverage on NERT

Thin phases; each ends runnable with its acceptance. Branch from `main`;
merge only when `validation.md` is fully evidenced.

## Phase A — Tier 4 RAFT (P4a, adapted)

- [T-1] `InstanceNormalization` kernel + shape rule + loader entry
  (`src/kernels/InstanceNormalization.cpp`, `Kernels.hpp`,
  `CpuKernels.cpp`, `CMakeLists.txt` source list, `ModelLoader.cpp`
  allowlist/policy, `ShapeInference.cpp` cases) + hand-computed tests
  (incl. non-default epsilon, scale/bias, negative case).
- [T-2] Close further RAFT gaps generally (survey says the other tier-4
  needs — AveragePool, GridSample, Pad, Range, ReduceSum, Sqrt, Tanh, Tile —
  already exist; anything new follows the same file pattern).
- [T-3] NERT-side proof: full `ctest` green; RAFT parity evidence recorded
  from Neuriplo (`tier4_raft.manifest`, real pair, atol 1e-3). If the harness
  needs the multi-file `;` input column, report it to Neuriplo — do not fork
  the harness here.
- Acceptance: `ctest --test-dir build` with `NERT_FIXTURE` (immutable
  acceptance) + RAFT row evidence in `validation.md`.

## Phase B — Tier 5 open vocabulary

- [T-4] Kernels + shape rules + loader entries + unit tests for `ArgMax`,
  `Elu`, `LessOrEqual`, `ReduceL2`, `CumSum`, `EyeLike`, `GreaterOrEqual`,
  `Max`, `Min`, `Or`, `ScatterND` (same file pattern as T-1; int64 compute
  where the exports need it).
- [T-5] OWLv2 + Grounding DINO end to end: load + manifest-row parity
  evidence from Neuriplo (int64 `input_ids`/`attention_mask` fill per
  manifest token; exact compare for int64/bool outputs).
- [T-6] Full regression: ResNet-18 + YOLO + transformer + DETR manifests
  green (recorded, not re-implemented).
- Acceptance: immutable acceptance + all Phase 3 rows evidenced.

## Phase C — Close-out

- [T-7] `CHANGELOG.md` entry (0.1.0 Unreleased/Added: new ops + families);
  roadmap Phase 3 → Complete; validation evidence + dates; deviations noted.
- [T-8] Report Phase 4 inputs: [Q-1]/[Q-3]/[Q-4] status, any harness gaps
  found, no speculative packets.

## Notes

- Phase order is load-bearing (A before B; C last). Kernel house style:
  nearest existing unary/broadcast/reduction kernel; comments match
  surrounding density; clang-format clean for touched files.
- Record deviations here with reasons.
