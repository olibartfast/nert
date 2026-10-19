# Catalog program — orchestration ledger

Date opened: 2026-10-19. Supersedes the P0-only framing; the release goal is the
complete Neuriplo tasks catalog. This file is the orchestrator's durable ledger.
The measured inventory is [`inventory.md`](inventory.md). The living device matrix
remains `specs/2026-10-17-task-models/coverage.md`, owned by the integration packet.

## Contract

- **Intent**: every `TaskFactory` model type runs end-to-end on CPU, CUDA and
  OpenCL against a reference, or carries an evidenced unsupported/blocked decision.
- **Acceptance scoreboard (per packet)**: the packet's single command, run once.
  Kernel packets: `cmake --build build-ocl -j8 && cd build-ocl && ctest --output-on-failure && cd .. && ./scripts/format.sh`.
  Integration packets additionally update `coverage.md` / `CHANGELOG.md`.
- **Roles**: planner (decomposition) → fresh implementer (one packet, named paths) →
  reviewer (read-only) → orchestrator (judge, dispatch defects).
- **Harness intervention (2026-10-19)**: the `implementer` agent was capped at
  `steps: 12`, which made every packet (read + write + build + test + acceptance)
  unachievable; two attempts died after read-only investigation. Raised to
  `steps: 40` in `~/.config/opencode/agents/implementer.md`. Recorded as a harness
  change, not a worker change.

## Baseline

`main` at `b38dd0e`; clean tree; CPU 12/12, CUDA 24/24, OpenCL 19/19.
Catalog: 32 entries — 2 fully covered (ResNet-18 gated export, OWLv2),
3 CPU+CUDA with OpenCL gaps (YOLO26n, RF-DETR seg-small, RAFT-large),
9 OPS-only, 1 blocked loader (Grounding DINO int64 Clip), 16 missing export,
1 out of scope (GGUF/llama.cpp). Counts and per-entry evidence: `inventory.md`.

## Run ledger

| # | Packet | Role / model | Attempt | Writable paths | Acceptance | Result | Intervention |
|---|---|---|---|---|---|---|---|
| 1 | P0a-1 math unary rows | implementer / zai-glm-5.3-flash | 1 | 2 new files | build-ocl + ctest + format | FAIL (step ceiling, 0 files) | re-dispatch; harness step cap raised 12→40 |
| 2 | P0a-1 math unary rows | implementer / zai-glm-5.3-flash | 2 | 2 new files | same | FAIL (step ceiling, 0 files) | prescriptive packet with settled design |
| 3 | P0a-1 math unary rows | implementer / zai-glm-5.3-flash | 3 | 2 new files | same | PASS 19/19, scratch 6/6 | accepted |
| 4 | Catalog inventory | explore / (read-only) | 1 | none | n/a | PASS, 32-entry matrix | accepted as `inventory.md` |
| 5 | P0a-2 reduce rows | implementer | 1 | 2 new files | same | PASS 19/19, scratch 6/6 | accepted |
| 6 | P0a-3 index/move rows | implementer | 1 | 2 new files | same | PASS 19/19, scratch 8/8 | accepted |
| 7 | P0a-7a wiring | implementer | 1 | table.cpp, CMakeLists.txt, test/CMakeLists.txt | 22/22 | FAIL 20/22 (stale `Sin` negatives) | integration defect, re-dispatch |
| 8 | P0a-7b corrected wiring | implementer | 1 | 2 test files | 22/22 | PASS 22/22, format clean | accepted |
| 9 | P0 catalog rows + gates | integration | 1 | 18 new rows + table/CMake/tests | build-ocl + ctest + format | PASS (suite green); YOLO26n, RF-DETR and RAFT gates pass | accepted |
| 10 | Catalog completion rows | integration | 1 | 13 new rows + table/CMake/tests | build-ocl + ctest + format | PASS 28/28; all loadable OpenCL op gaps 0 | accepted |

## Shared OpenCL rows

Implemented once, reused across models:

- P0a-1: `Sqrt`, `Cos`, `Sin`, `Exp`, `Not`, `Less` — `src/opencl/math_unary_extra.cpp`
- P0a-2: `ReduceSum`, `ReduceMax` — `src/opencl/reduce_extra.cpp`
- P0a-3: `Split`, `GatherElements`, `TopK`, `Mod`, `Tile` — `src/opencl/index_move.cpp`

No shared OpenCL rows are pending for the loadable catalog. Remaining
work is full-model gates for converted variants, dtype/shape variants,
and the HIP decision (no ROCm toolchain or AMD hardware).

## Context-reset checkpoints

Each model gate is run by a fresh worker after its rows are committed; the
orchestrator retains only this ledger, `inventory.md`, and the per-gate evidence
rows in `coverage.md` / `validation.md`. No implementation detail is carried into
the next model's session.
