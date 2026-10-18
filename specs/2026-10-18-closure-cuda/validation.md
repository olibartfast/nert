# Validation — CUDA closure: GridSample + ConvTranspose + InstanceNormalization

Defined before implementation. Tolerance: `ExpectNear` bench rule
unless exact. SKIP without device/cuDNN is recorded.

## Checks

- [V-A] GridSample: bilinear hand values + zeros/border/reflection
  edges + nearest half-even + align_corners, CPU vs CUDA; negatives
  (mode, padding_mode, batch mismatch, non-4D).
- [V-B] InstanceNorm: hand values, rank-2 input, per-channel scale/B;
  negatives (scale size, rank<2, dtype).
- [V-C] ConvTranspose (cuDNN leg): stride-2 hand-computed, padded+bias
  vs oracle within tolerance; negatives (output shape, output_padding
  reject, asymmetric-pad reject). CPU-only leg unaffected (row absent
  without cuDNN).
- [V-D1] OFF leg full `ctest` green. [V-D2] CUDA-ON leg full `ctest`
  green (RTX 3060). [V-D3] cuDNN leg (`CUDNN_ROOT` farm) full `ctest`
  green. [V-D4] format clean. [V-D5] diff audit: packet paths only.

## Acceptance (once at the end)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel "$(nproc)" && ctest --test-dir build --output-on-failure && ./scripts/format.sh --check
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DNERT_ENABLE_CUDA=ON && cmake --build build-cuda --parallel "$(nproc)" && ctest --test-dir build-cuda --output-on-failure
CUDNN_ROOT=/tmp/opencode/cudnn cmake -S . -B build-cudnn -DCMAKE_BUILD_TYPE=Release -DNERT_ENABLE_CUDA=ON && cmake --build build-cudnn --parallel "$(nproc)" && LD_LIBRARY_PATH=/home/oli/.local/lib/python3.12/site-packages/nvidia/cudnn/lib ctest --test-dir build-cudnn --output-on-failure
```

Host prerequisite: a bare `ctest` in `build-cudnn` aborts on this host
with `Cannot load symbol cublasLtGetVersion` unless `LD_LIBRARY_PATH`
points at a CUDA-12-matched pip cuDNN (above) or at
`$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib:/tmp/opencode/cudnn/lib64`.
The latter is the CUDA-13 runtime + 9.25 farm; both were verified 24/24.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-A | `nert_cuda_sample_test` bilinear/border/reflect/nearest/align + 2 negatives | pass 7/7 | 2026-10-18 | New target; half-even + align hand values |
| V-B | `nert_cuda_norm_test` InstanceNorm exact/rank-2/negative | pass 3/3 new | 2026-10-18 | Suite 9/9; per-(n,c) double-accum rows |
| V-C | `nert_cuda_conv_test` ConvTranspose stride-2/bias/negative | pass 3/3 new | 2026-10-18 | cuDNN farm leg; bwd-data + first-party bias |
| V-D1 | OFF leg full `ctest` | pass 12/12 | 2026-10-18 | Fixture fallback |
| V-D2 | CUDA-ON (no cuDNN) full `ctest` | pass 24/24 | 2026-10-18 | RTX 3060; probe relocated (see deviations) |
| V-D3 | CUDA-ON cuDNN farm full `ctest` | pass 24/24 | 2026-10-18 | `CUDNN_ROOT` pip-9 farm; probe skips, CT runs |
| V-D4 | `scripts/format.sh --check` | pass | 2026-10-18 | `--fix` on new tests only |
| V-D5 | Diff audit | pass | 2026-10-18 | Packet paths + 2 cross-session repairs (below) |

## Deviations

- `implementer` dispatch unavailable (harness); planner implements directly.
- Cross-session repair 1: `src/cuda/table.cu` carried a broken guard
  since `4b4a962` — first-party `Einsum`'s decl sat inside
  `#ifdef NERT_HAS_CUDNN` while its row (and unconditional compile)
  sat outside, breaking every non-cuDNN CUDA build. Moved the decl
  out (one line); first-party rows must never sit in toolkit guards.
- Cross-session repair 2: `UnknownOpRejectsNamingNodeAndOp` probed
  InstanceNormalization ("never needs moving again"); CUDA closure
  gave it a row. Probe moved to ConvTranspose — the last
  device-conditional row — with a runtime row check that skips on
  cuDNN builds, plus a comment stating the next move.
