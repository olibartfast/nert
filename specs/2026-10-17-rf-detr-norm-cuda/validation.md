# Validation — RF-DETR norm corridor on CUDA (R1–R3 + ReduceSum)

Defined before implementation. Tolerance: `ExpectNear` bench rule
(`1e-3 * (1 + |ref|)` elementwise) unless a check states exact match.
GPU-gated tests SKIP without a device — a SKIP is recorded, never
re-run as a failure.

## Checks

- [V-A1] Sqrt: hand-computed (incl. NaN for negative input, exact zeros)
  on CPU and CUDA agree.
- [V-A2] Erf: hand-computed (`erf(0)=0`, `erf(1)≈0.84270079`,
  odd symmetry) on CPU and CUDA agree.
- [V-A3] Sqrt/Erf negatives: count mismatch and non-float32 reject,
  naming node + op.
- [V-B1] ReduceSum: hand-computed (axes subset, `keepdims` 0/1, absent
  axes = all axes, `noop_with_empty_axes` identity incl. shape check,
  negative axes, int64 data) CPU vs CUDA agree.
- [V-B2] ReduceSum negatives: bad axis, duplicate axis, bad output
  shape, non-int64 axes reject, naming node + op.
- [V-C1] LayerNormalization: hand-computed (default axis -1, axis=1,
  scalar Scale/B, missing B, Mean/InvStdDev outputs, epsilon default)
  CPU vs CUDA agree.
- [V-C2] LayerNormalization negatives: bad axis, bad Scale size,
  Y-shape mismatch reject, naming node + op.
- [V-D1] OFF leg: default configure + build + full `ctest` green
  (fixture via `/tmp/opencode/resnet18_export/resnet18.onnx` fallback).
- [V-D2] CUDA-ON leg: `-DNERT_ENABLE_CUDA=ON` configure + build + full
  `ctest` green on the RTX 3060.
- [V-D3] `scripts/format.sh --check` clean.
- [V-D4] Diff audit: only `src/cuda/sqrt.cu`, `src/cuda/erf.cu`,
  `src/cuda/reduce_sum.cu`, `src/cuda/layer_normalization.cu`,
  `src/cuda/table.cu`, `CMakeLists.txt` (sources only),
  `test/cuda_activations_test.cpp`, `test/cuda_misc_test.cpp`,
  `test/cuda_norm_test.cpp`, `test/CMakeLists.txt`, and this packet
  changed.

## Acceptance (run once, at the end of each phase attempt)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel "$(nproc)" && ctest --test-dir build --output-on-failure && ./scripts/format.sh --check
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DNERT_ENABLE_CUDA=ON && cmake --build build-cuda --parallel "$(nproc)" && ctest --test-dir build-cuda --output-on-failure
```

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-A1 | `nert_cuda_activations_test` SqrtExact (NaN + zeros) | pass | 2026-10-17 | RTX 3060; CPU+CUDA vs hand values |
| V-A2 | `nert_cuda_activations_test` ErfExact + symmetry | pass | 2026-10-17 | erf(1)≈0.84270079, erf(0.5)≈0.52049988 |
| V-A3 | SqrtRejectsCountMismatch, ErfRejectsNonFloat | pass | 2026-10-17 | Messages name node + op |
| V-B1 | `nert_cuda_misc_test` ReduceSum axis/int64/noop-identity | pass | 2026-10-17 | float32 double-accum + int64 paths |
| V-B2 | ReduceSum dup-axis/bad-axis/non-int64-axes rejects | pass | 2026-10-17 | Messages name node + op |
| V-C1 | `nert_cuda_norm_test` default-axis/scale-bias-stats/axis-0 | pass | 2026-10-17 | Double-accum rows; Mean/InvStdDev checked |
| V-C2 | LayerNorm bad-axis/bad-Scale/Y-mismatch rejects | pass | 2026-10-17 | Messages name node + op |
| V-D1 | OFF leg: default configure + build + `ctest` | pass 12/12 | 2026-10-17 | Fixture via `/tmp/opencode/resnet18_export` fallback |
| V-D2 | CUDA-ON leg: `-DNERT_ENABLE_CUDA=ON` + full `ctest` | pass 22/22 | 2026-10-17 | RTX 3060 host leg |
| V-D3 | `scripts/format.sh --check` | pass | 2026-10-17 | `--fix` applied to line wraps in new tests only |
| V-D4 | Diff audit | pass | 2026-10-17 | Slice paths only; pre-existing dirty (`README.md`, `bench/`) untouched |

## Deviations

- Delegation harness failure: two `implementer` subagent dispatches errored
  (`tool_choice` unsupported), but the first attempt's file writes landed
  before the failure (Phase A kernels/table/CMake/tests). Duplicate table
  rows from the retry were removed; Phases B/C implemented directly by the
  planner. No silent self-repair: all code follows the packet's stated
  patterns and is covered by the checks above.
- `ReduceSumNoopIdentity` first failed on a test bug (0-byte axes upload);
  fixed by passing a null view — the kernel already guards empty axes.
  Kernel code unchanged.
- Work done in the main tree per maintainer direction (deviation from the
  clean-worktree default, recorded in `plan.md`).
