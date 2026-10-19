# Catalog program — orchestration ledger

Opened 2026-10-19. The release goal is the complete Neuriplo tasks catalog
(30 in-scope ONNX entries of 32; image understanding and Gaussian Splatting are
maintainer-approved OUT OF SCOPE). Measured inventory: [`inventory.md`](inventory.md).
Living device matrix: `specs/2026-10-17-task-models/coverage.md` (integration owner only).

## Contract

- **Acceptance scoreboard (per packet)**: the packet's single command, run once, as
  the final action. Kernel packets: `cmake --build build-ocl -j8 && cd build-ocl && ctest --output-on-failure && cd .. && ./scripts/format.sh`.
  Integration packets additionally update `coverage.md` / `CHANGELOG.md`.
- **Roles**: planner (decomposition) → fresh implementer (one packet, named paths) →
  **fresh read-only reviewer** (mandatory before a packet is accepted) → orchestrator
  (judge, dispatch defects). Gate runners (`general`) are verification, not a reviewer.
- **Concurrency**: hard cap of 2 concurrently running packets; shared files
  (`src/opencl/table.cpp`, `CMakeLists.txt`, `test/CMakeLists.txt`,
  `specs/2026-10-17-task-models/coverage.md`, `CHANGELOG.md`) belong to one
  serialized integration owner.
- **Harness interventions (2026-10-19)**: `implementer` agent step cap raised 12 → 40
  (`~/.config/opencode/agents/implementer.md`) because no packet could complete read +
  write + build + test + acceptance in 12 steps; `reviewer` agent model repointed from
  `deepseek/deepseek-flash` (insufficient balance) to `zai/glm-5.3-flash`.

## Run ledger

Orchestrator session for rows 1–26: `ses_eaca1c986ffeFJdfdPOFfK3mDh`.

| # | Packet | Role / model | Attempt | Writable paths | Concurrency at start | Acceptance | Result | Reviewer session | Reviewer verdict | Stop condition |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | P0a-1 math unary rows | implementer | 1 | 2 new files | 1 | build-ocl+ctest+format | FAIL (step ceiling, 0 files) | — | — | stop after acceptance; no repair after it |
| 2 | P0a-1 retry | implementer | 2 | 2 new files | 1 | same | FAIL (step ceiling, 0 files) | — | — | same |
| 3 | P0a-1 retry | implementer | 3 | 2 new files | 1 | same | PASS 19/19, scratch 6/6 | (pre-reviewer era) | — | same |
| 4 | Catalog inventory | explore | 1 | none (read-only) | 1 | n/a | PASS, 32-entry matrix | — | — | read-only, no repo writes |
| 5 | P0a-2 reduce rows | implementer | 1 | 2 new files | 2 | same | PASS 19/19, scratch 6/6 | — | — | same |
| 6 | P0a-3 index/move rows | implementer | 1 | 2 new files | 2 | same | PASS 19/19, scratch 8/8 | — | — | same |
| 7 | P0a-7a wiring | implementer | 1 | table.cpp, 2 CMake files | 1 | 22/22 | FAIL 20/22 (stale `Sin` negatives) | — | — | report the failure, do not edit other owners' files |
| 8 | P0a-7b corrected wiring | implementer | 1 | 2 test files | 1 | 22/22 | PASS 22/22 | — | — | same |
| 9 | P0a-4 Resize | implementer | 1 | 2 new files | 2 | build-ocl+ctest+format | PASS 19/19, scratch 6/6 | — | — | stop after acceptance |
| 10 | P0a-5 GridSample | implementer | 1 | 2 new files | 2 | same | PASS 19/19, scratch 7/7 | — | — | same |
| 11 | P0a-7c wiring | implementer | 1 | table.cpp, 2 CMake files | 1 | 24/24 | PASS | — | — | same |
| 12 | P0a-6 norm/range/pad | implementer | 1 | 2 new files | 2 | same | PASS 19/19, scratch 8/8 | — | — | same |
| 13 | P0e-1 int64 Clip CPU/CUDA | implementer | 1 | clip.cpp, clip.cu, shape_inference.cpp, 2 test files | 2 | CPU 12/12 + CUDA 24/24 + format | PASS | — | — | stop after acceptance |
| 14 | P0a-7d wiring | implementer | 1 | table.cpp, 2 CMake files | 1 | 25/25 | PASS | — | — | same |
| 15 | P0-D1 docs | implementer | 1 | coverage.md, CHANGELOG.md | 1 | diff stat + format | PASS (55 rows) | — | — | docs only, no code |
| 16 | Commit + push (orchestrator) | orchestrator | 1 | — | 1 | CPU 12/12, CUDA 24/24, OpenCL 25/25, format | PASS → commit `6d99da9` pushed | — | — | commit scoped paths only |
| 17 | P0a-2b int64 ReduceSum | implementer | 1 | reduce_extra.cpp + test | 1 | 25/25 | PASS (found by the RF-DETR gate) | — | — | stop after acceptance |
| 18 | P0a-8 elementwise extras | implementer | 1 | 2 new files | 2 | same | PASS, scratch 7/7 | — | — | same |
| 19 | P0a-9 index/scan extras | implementer | 1 | 2 new files | 2 | same | PASS, scratch 8/8 | — | — | same |
| 20 | P0a-7e wiring | implementer | 1 | table.cpp, 2 CMake files | 1 | 27/27 | PASS | — | — | same |
| 21 | P0a-10 ConvTranspose | implementer | 1 | 2 new files | 1 | 27/27 | PASS, scratch 6/6 | — | — | same |
| 22 | P0a-7f wiring | implementer | 1 | table.cpp, 2 CMake files | 1 | 28/28 | PASS | — | — | same |
| 23 | Gate: YOLO26n / RF-DETR seg-small / RAFT-large OpenCL | general | 1 | /tmp only | 2 | gate evidence | PASS / PASS / PASS | — | — | report a missing kernel exactly; no kernel edits |
| 24 | Review of `b38dd0e..6d99da9` | reviewer | 1 | read-only | 1 | n/a | — | `ses_eab6d9dfdffeSofT0ScKBzaLpe` | PASS-WITH-FINDINGS (2 major test gaps, minors) | read-only; no writes |
| 25 | Review of the 13 uncommitted rows (now `b905792`) | reviewer | 1 | read-only | 1 | n/a | — | `ses_eab5e9a8effeG6SJ8Gw8nLibl4` | **REJECT** (1 blocker: ScatterND duplicate ordering; 1 major: `Or` arity) | read-only; no writes |
| 26 | FIX-1 defect packet | implementer | 1 | index_extra.cpp, math_extra2.cpp, 2 test files | 1 | targeted run + 28/28 | edits written; acceptance NOT run (step ceiling) | — | — | do not re-run acceptance; report state |
| 27 | FIX-1b verification | implementer | 1 | same 4 files (+shape_inference.cpp) | 1 | CPU 12/12 + OpenCL 28/28 + format | PASS (21/21 targeted) | — | — | single acceptance run, no commit |
| 28 | Review of FIX-1 | reviewer | 1 | read-only | 1 | n/a | — | `ses_eab4021ddffeejupEAimDqzfQk` | PASS-WITH-FINDINGS (all minor) | read-only |
| 29 | INT-2 docs integration | implementer | 1 | coverage.md, CHANGELOG.md | 1 | diff stat + status | PASS (68 rows, 13/30 scoreboard) | — | — | docs only; no commit by the worker |
| 30 | Commit + push (orchestrator) | orchestrator | 1 | 7 scoped paths | 1 | OpenCL 28/28, CPU 12/12, format | PASS → commit `346cb29` pushed | — | — | commit scoped paths only |

## Commits

| Commit | Content |
|---|---|
| `6d99da9` | 18 OpenCL rows (37 → 55), YOLO26n / RF-DETR seg-small / RAFT-large OpenCL gates, int64 Clip CPU+CUDA |
| `b905792` | 13 more OpenCL rows (55 → 68) + coverage/ledger/CHANGELOG (committed by the maintainer during the review window) |
| `346cb29` | FIX-1: ScatterND duplicate ordering, `Or` arity, ReduceMean noop convention, Identity/Squeeze rank cap; catalog gate evidence; HIP unsupported |

## Model boundaries (kernel + gate + reviewer + commit)

| Model | Commit | Rows added | Reviewer | Gate evidence | Next model |
|---|---|---|---|---|---|
| YOLO26n | `6d99da9` | Split, GatherElements, Resize, TopK, Mod, ReduceMax | (pre-reviewer; covered by `ses_eab6d9dfdffeSofT0ScKBzaLpe`) | OpenCL maxdiff 4.88e-04, 0/1800 over tol, min IoU 0.9999984 | RF-DETR seg-small |
| RF-DETR seg-small | `6d99da9` | Sqrt, ReduceSum, Cos, Sin, Exp, Not, Less, Tile, GridSample | same | dets 6.82e-04, labels 100/100, mask IoU min 0.9996687 | RAFT-large |
| RAFT-large | `6d99da9` | GridSample, InstanceNormalization, Range, Pad | same | flow 1.32e-04 vs ORT, 0/365,056 over 5.324e-03 | Grounding DINO |
| Catalog variants (YOLO11s, YOLOv8n-pose, YOLO26s-pose, YOLO26s, YOLO26x, RF-DETR nano-1101, YOLO26n-depth, RF-DETR keypoint-preview) | `346cb29` | Clip, Log, Pow, Max, Min, Or, Identity, Squeeze, CumSum, EyeLike, ScatterND, ReduceMean, ConvTranspose | `ses_eab4021ddffeejupEAimDqzfQk` (FIX-1), `ses_eab5e9a8effeG6SJ8Gw8nLibl4` (rows) | per-model metrics in coverage.md; YOLO26s/x carry the R-6 tie-flip deviation | Grounding DINO gate |

## Context-reset checkpoints

| Checkpoint | New orchestrator session | When |
|---|---|---|
| After the 13-row catalog boundary (commit `346cb29`) | `ses_eab7c07feffe0Lf0lMbP1D7gFb` (dispatched as agent `orchestrator`) | end of session `ses_eaca1c986ffeFJdfdPOFfK3mDh` |

Rule applied: at a model boundary the orchestrator writes the handoff, stops the
implementing worker, and starts a fresh orchestrator session that reads only the
handoff, this ledger, the inventory and the acceptance commands. No implementation
detail is carried into the new session.
