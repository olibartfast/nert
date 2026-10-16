# Orchestration — Tier-5 → Phase 4 → Phase 5 (incl. filenames) → N1

Date: 2026-10-16. Base: `main@649a300` (bench gemm+matmul merged, Tier-4 RAFT done).
Constitution: `specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md`.
Base packet for Phase B: `specs/2026-10-10-raft-open-vocab/` (R-2/R-4, T-4/T-5, V-1/V-4/V-5 open).

## Scope to N1

- **B — Tier-5 open-vocab** (open): 11 ops `ArgMax/Elu/LessOrEqual/ReduceL2/CumSum/EyeLike/GreaterOrEqual/Max/Min/Or/ScatterND` + OWLv2/Grounding-DINO parity. Sliced thin for worker context:
  - B1: `Elu/LessOrEqual/GreaterOrEqual/Or/Max/Min` (unary/binary, template: `Less.cpp/Greater.cpp/Equal.cpp/And.cpp`, `Sigmoid.cpp` for Elu)
  - B2: `ArgMax/ReduceL2/CumSum/EyeLike` (reduction/creation, template: `ReduceMean/ReduceSum/TopK/ConstantOfShape/Range`)
  - B3: `ScatterND` + OWLv2/Grounding-DINO load+parity + full regression (YOLO/transformer/DETR)
- **Phase 4 — Hardening+Policy**: [Q-1] opset policy, [Q-3] static-per-load close, [Q-4] CI zoo subset, T-10 close-out, CHANGELOG. No new ops.
- **Phase 5 — Breaking 0.2 rename (blocked on maintainer, now scheduled)**:
  - [Q-5] `include/engine/` + namespace `engine` → `nert` (+ Neuriplo adapter note, not implemented here)
  - **Filenames → modern C++**: all `.hpp`/`.cpp`, lowercase `snake_case`, no PascalCase:
    `src/kernels/InstanceNormalization.cpp` → `instance_normalization.cpp`, `GatherElements.cpp` → `gather_elements.cpp`, `src/ModelLoader.cpp` → `model_loader.cpp`, `include/engine/ModelLoader.hpp` → `include/nert/model_loader.hpp`, etc. `bench/` (`gemm.hpp`, `matmul.hpp`, `benchmark.hpp`) already compliant — reference style. One breaking commit, never mixed into op work. CMake source lists + `#include` paths updated mechanically. `scripts/format.sh` scope updated.
- **N1 — CUDA device layer**: same model+plan on CUDA matching CPU oracle within tolerance, one device per graph at load. Blocked on [Q-1] vendor-kernel decision (first-party tiled vs opt-in cuBLAS/cuDNN) — decided in Phase 4, implemented here. Reuses `bench/` tiled kernels as measurement reference, not pasted adapters (per bench [D-1]: shared harness header + oracle source of truth + kernel-table adapters).

## Roles / boundaries (from `2026-10-10-raft-open-vocab/orchestration.md`, held)

- Planner/orchestrator: main session (strongest) — packets, acceptance, integration, commits, ledger.
- Implementer: subagent, one phase at a time, fresh context, mid tier — packet writable paths only.
- Reviewer (optional): read-only, never a diff.
- Enforced: each implementer in own worktree `../nert-w-<phase>` (`wip/nert-<phase>` from packet start rev); reviewer no write; planner-only commits.
- Advisory: writable-path lists, never-commit rule (planner checks `git status`/`diff --stat`).
- Limits: builds `--parallel 2`; one implementer at a time.
- Parity runs in Neuriplo: planner-run after integration, never worker acceptance. Worker acceptance = immutable ResNet fixture ctest only.

## Acceptance (planner-owned; workers run once, final action)

```bash
cmake -S . -B build && cmake --build build --parallel 2 \
 && NERT_FIXTURE=fixtures/resnet18.onnx ctest --test-dir build --output-on-failure \
 && scripts/format.sh --check && git status --short
```

## Skill map — https://github.com/olibartfast/skills (checked 2026-10-16)

- Workflow: `orchestrate-ai-coding-workflows` + `apply-spec-driven-development` (this run).
- B1–B3 kernels/tests: `test-cpp-software` (hand-computed cases + negative load/shape), `modernize-cpp-design` (closed op sets, error model).
- Phase 5 rename + filenames: `modernize-cpp-design` + `engineer-modern-cmake` (target graph, CMake source lists, `scripts/format.sh` scope). Rule: lowercase `snake_case` `.hpp`/`.cpp` only; `bench/` is the exemplar.
- N1 CUDA: `select-cuda-parallel-patterns` + `optimize-cuda-memory-locality` + `coordinate-cuda-thread-groups` + `pipeline-cuda-streams` + `profile-optimize-cuda` + `implement-cuda-data-primitives` (ScatterND/TopK-adjacent scatter/gather review).

## Compact-after-each-step

After each slice: `rm -rf build build-* /tmp/opencode/*scratch*` in worktree + main as applicable, subagent summarizes (<30 lines: files touched, tests, deltas), planner appends ledger row + evidence dates. No carry-over context into next slice.

## Run ledger

| Attempt | Packet slice | Worktree/branch | Acceptance (first pass) | Integration gate | Notes |
|---|---|---|---|---|---|
| 1 | B1 6-op (Elu/LE/GE/Or/Max/Min) | `../nert-w-b1` / `wip/nert-b1` | not run (recon budget, 0 writes) | — | fail: slice too wide; split to B1a/B1b |
| 2 | B1a Elu/Max/Min | `../nert-w-b1` / `wip/nert-b1` | 8/8 non-fixture pass; 4 fixture suites fail missing-fixture by design; format clean | merged 249681b | done 2026-10-16: planner-implemented after 2 worker recon stalls; scope 9 files, +577/-3 |
| 3 | B1b LessOrEqual/GreaterOrEqual/Or | `../nert-w-b1b` / `wip/nert-b1b` | 8/8 non-fixture pass; format clean | merged a9b232d | done 2026-10-16: direct, +553/-6 |
| 4 | B2a ReduceL2 | `../nert-w-b2` / `wip/nert-b2` | 8/8 non-fixture pass; format clean | merged f4b5cf2 | done 2026-10-16: direct, incl. double-accum proof |
| — | B2b ArgMax/CumSum/EyeLike | `../nert-w-b2b` / `wip/nert-b2b` | — | — | next |
| — | B3 ScatterND+e2e | `../nert-w-b3` / `wip/nert-b3` | — | — | queued |
| — | Phase 4 hardening | packet to be written | — | — | queued |
| — | Phase 5 rename+filenames | one breaking commit | — | — | queued, maintainer gate |
| — | N1 CUDA layer | packet after Q-1 decision | — | — | queued |
