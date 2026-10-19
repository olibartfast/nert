# Validation — CUDA Operator Closure

Defined before implementation. The comparison uses the benchmark's
`ExpectNear` tolerance unless a check is exact. A missing CUDA device or cuDNN
installation records `SKIP` with the reason.

## What Each Check Proves

**GridSample.** Hand-computed bilinear values, edge behavior for zeros,
border, and reflection padding, nearest sampling with half-even rounding, and
`align_corners` are compared between CPU and CUDA. Negative cases cover an
invalid mode, an invalid padding mode, a batch mismatch, and a non-4D input.

**InstanceNormalization.** Hand-computed values, a rank-2 input, and
per-channel scale and bias are compared with the CPU oracle. Negative cases
cover a bad scale size, a rank below 2, and a mismatched dtype.

**ConvTranspose.** On the cuDNN build, a stride-2 case is checked against a
hand-computed result and a padded case with bias is compared with the oracle.
Negative cases cover a bad output shape, non-zero `output_padding`, and
asymmetric padding. The CPU-only build is unaffected because the row is absent
without cuDNN.

**Build matrix.** The default build, the CUDA build without cuDNN, and the
CUDA build with cuDNN must all pass their full CTest suites. Formatting must be
clean, and the diff must stay within the expected specification and source
paths.

## Acceptance Commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --parallel "$(nproc)" && \
    ctest --test-dir build --output-on-failure && \
    ./scripts/format.sh --check

cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
    -DNERT_ENABLE_CUDA=ON && \
    cmake --build build-cuda --parallel "$(nproc)" && \
    ctest --test-dir build-cuda --output-on-failure

CUDNN_ROOT=/tmp/opencode/cudnn \
    cmake -S . -B build-cudnn -DCMAKE_BUILD_TYPE=Release \
    -DNERT_ENABLE_CUDA=ON && \
    cmake --build build-cudnn --parallel "$(nproc)" && \
    LD_LIBRARY_PATH=${HOME}/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
    ctest --test-dir build-cudnn --output-on-failure
```

## Host Prerequisite

A bare `ctest` in `build-cudnn` aborts on this host with
`Cannot load symbol cublasLtGetVersion` unless `LD_LIBRARY_PATH` points at a
CUDA-12-matched pip cuDNN, as in the command above, or at
`$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib:/tmp/opencode/cudnn/lib64`.
The second path is the CUDA-13 runtime with the 9.25 cuDNN farm. Both paths
were verified with 24 of 24 tests passing.

## Evidence

| ID | Check | Result | Date | Notes |
| --- | --- | --- | --- | --- |
| `V-A` | `nert_cuda_sample_test`: bilinear, border, reflection, nearest, `align_corners`, plus two negative cases | Pass, 7/7 | 2026-10-18 | New test target. Half-even and `align_corners` values are hand-computed. |
| `V-B` | `nert_cuda_norm_test`: InstanceNormalization exact values, rank-2 input, negative case | Pass, 3/3 new | 2026-10-18 | Full suite 9/9. Per-`(n, c)` rows use double accumulation. |
| `V-C` | `nert_cuda_conv_test`: ConvTranspose stride-2, bias, negative case | Pass, 3/3 new | 2026-10-18 | cuDNN farm build. Uses backward-data plus the first-party bias add. |
| `V-D1` | Default build full CTest | Pass, 12/12 | 2026-10-18 | Fixture fallback. |
| `V-D2` | CUDA build without cuDNN, full CTest | Pass, 24/24 | 2026-10-18 | RTX 3060. The unknown-operator probe moved; see deviations. |
| `V-D3` | CUDA build with the cuDNN farm, full CTest | Pass, 24/24 | 2026-10-18 | `CUDNN_ROOT` pip-9 farm. The probe skips and CTest runs. |
| `V-D4` | `scripts/format.sh --check` | Pass | 2026-10-18 | `--fix` applied only to new tests. |
| `V-D5` | Diff audit | Pass | 2026-10-18 | Specification paths plus two cross-session repairs, listed below. |

## Deviations

- The implementer dispatch was unavailable in the harness, so the planner
  implemented the work directly.
- **Cross-session repair 1.** `src/cuda/table.cu` had carried a broken guard
  since commit `4b4a962`: the declaration for the first-party `Einsum` row sat
  inside `#ifdef NERT_HAS_CUDNN`, while its table entry and unconditional
  compilation sat outside. This broke every non-cuDNN CUDA build. The
  declaration was moved out. First-party rows must never sit inside toolkit
  guards.
- **Cross-session repair 2.** The `UnknownOpRejectsNamingNodeAndOp` test used
  `InstanceNormalization` as its unknown-operator probe, but this closure gave
  that operator a CUDA row. The probe moved to `ConvTranspose`, the last
  device-conditional row. It now checks the row at runtime and skips on cuDNN
  builds, with a comment explaining where it should move next.
