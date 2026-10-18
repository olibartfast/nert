# Validation — OpenCL pools

Defined before implementation. Tolerance: `ExpectNear` bench rule
unless exact. SKIP without an OpenCL GPU platform is recorded.

## Checks

- [V-P1] MaxPool k2/s2 + pads + ceil_mode + dilation vs oracle.
- [V-P2] AveragePool count_include_pad 0/1 (incl. ceil-past-input
  divisor) vs oracle.
- [V-P3] GlobalAveragePool rank-4 + rank-3 vs oracle (double accum).
- [V-P4] Negatives: missing kernel_shape, bad output shape, bad pads.
- [V-D1] OFF leg full `ctest` green. [V-D2] CUDA-ON leg full `ctest`
  green. [V-D3] OpenCL-ON leg full `ctest` green (RTX 3060).
  [V-D4] format clean. [V-D5] diff audit: packet paths only.

## Acceptance (once at the end)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel "$(nproc)" && ctest --test-dir build --output-on-failure && ./scripts/format.sh --check
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DNERT_ENABLE_CUDA=ON && cmake --build build-cuda --parallel "$(nproc)" && ctest --test-dir build-cuda --output-on-failure
cmake -S . -B build-ocl -DCMAKE_BUILD_TYPE=Release -DNERT_ENABLE_OPENCL=ON && cmake --build build-ocl --parallel "$(nproc)" && ctest --test-dir build-ocl --output-on-failure
```

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-P1 | MaxPool k2/s2 + dilated/padded + negative | pass 3/3 new | 2026-10-18 | RTX 3060; -inf OOB, ceil, dilation green |
| V-P2 | AveragePool k2/s2 + count_include_pad 0/1 + negative | pass 3/3 new | 2026-10-18 | Test-dim slip + 24-vs-16B Download slip fixed (both oracle-right) |
| V-P3 | GAP rank-4 + negative | pass 2/2 new | 2026-10-18 | Double accum bit-matches oracle |
| V-D1 | OFF leg full `ctest` | pass 12/12 | 2026-10-18 | Fixture fallback |
| V-D2 | CUDA-ON leg full `ctest` | pass 24/24 | 2026-10-18 | Unchanged by this slice |
| V-D3 | OpenCL-ON leg full `ctest` | pass 14/14 | 2026-10-18 | Table 15→18 rows |
| V-D4 | `scripts/format.sh --check` | pass | 2026-10-18 | Post-fix rebuild rerun green |
| V-D5 | Diff audit + coverage.md sync | pass | 2026-10-18 | Packet paths + matrix recompute |

## Deviations

- `implementer` dispatch unavailable (harness); planner implements directly.
