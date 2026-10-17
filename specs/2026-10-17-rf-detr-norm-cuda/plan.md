# Plan — RF-DETR norm corridor on CUDA (R1–R3 + ReduceSum)

Work in the main tree (maintainer-directed; deviation from the
clean-worktree default recorded here). The tree is dirty
(`README.md`, `bench/runtime/*`, untracked
`specs/2026-10-17-cudnn-frontend-spike/`): those paths are forbidden —
never read-to-copy, never edit. Planner commits touch main only; code
lands on full validation evidence.

Three sequential phases (shared `src/cuda/table.cu` + `CMakeLists.txt`
force sequencing; easy → hard):

## Phase A — Sqrt + Erf (unary)

Pattern: `src/cuda/sigmoid.cu` (one thread per element, `CheckCuda`,
`ElementCount`, early return on zero count). Device math: `sqrtf`,
`erff`. Oracle: `src/kernels/sqrt.cpp`, `src/kernels/erf.cpp`
(float32-only, count match, null-buffer checks). Tests appended to
`test/cuda_activations_test.cpp` beside Sigmoid/Tanh.

## Phase B — ReduceSum (reduction)

Pattern: `src/cuda/reduce_max.cu` (per-output-element thread replaying
the oracle scan; axes download via `cudaMemcpy` D2H; `keepdims`; unique
normalized axes; rank limit 16; `IndexSet`/`Pack`). Additions over the
pattern: `noop_with_empty_axes` identity path (D2D copy, mirroring the
CPU oracle) and int64 support alongside float32. Oracle:
`src/kernels/reduce_sum.cpp`. Tests in `test/cuda_misc_test.cpp`
beside ReduceMax.

## Phase C — LayerNormalization (norm row)

Pattern: Softmax's one-thread-per-row with double accumulation (see
bench note) extended to mean + variance + scale + shift. Attributes
`axis` (default -1, negative wrap, range check) and `epsilon`
(default 1e-5). Scale/B per-element or scalar (`count == 1`); absent
optional B arrives as `DataType::Unknown` — skip, don't fail (CPU
precedent). Outputs 1–3: Y plus optional float32 Mean/InvStdDev with
one value per row. Oracle: `src/kernels/layer_normalization.cpp`.
New `test/cuda_norm_test.cpp` + `test/CMakeLists.txt` registration
(mirror an existing `nert_cuda_*` block).

## Per-phase bars (all three)

Hand-computed + oracle-compare + negatives; OFF 8/8 green; CUDA-ON leg
green on the RTX 3060 (SKIP recorded if no device); `format.sh --check`
clean; diff audit shows only the packet's allowed paths.

## Validation

Per-op evidence lands in
`specs/2026-10-17-rf-detr-norm-cuda/validation.md` — one table.
Roadmap + CHANGELOG sync on landing (planner-owned, not workers).
