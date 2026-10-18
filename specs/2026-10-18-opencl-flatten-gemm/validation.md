# Validation — OpenCL Flatten + Gemm

Defined before implementation. Tolerance: `ExpectNearOcl` unless exact;
`GTEST_SKIP` without an OpenCL GPU platform is recorded.

## Checks

- [V-F1] Flatten axis 0/1/-1 hand-computed + CPU-oracle compare;
  f32/int64/bool preserves bytes.
- [V-F2] Flatten negative: bad axis names node + op.
- [V-G1] Gemm 2x2 hand-computed + CPU-oracle compare.
- [V-G2] Gemm transA, transB, alpha/beta, bias `[N]` and `[M,N]`
  vs oracle.
- [V-G3] Gemm negatives: contraction mismatch, bad bias shape, wrong
  output shape.
- [V-D1] OFF leg full `ctest` green. [V-D2] CUDA-ON full `ctest` green.
  [V-D3] OpenCL-ON full `ctest` green (RTX 3060). [V-D4] format clean.
  [V-D5] diff audit: packet paths only.

## Acceptance

```bash
cmake --build build && ctest --test-dir build --output-on-failure
cmake --build build-cuda && ctest --test-dir build-cuda --output-on-failure
cmake --build build-ocl && ctest --test-dir build-ocl --output-on-failure
./scripts/format.sh --check
```

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-F1 | Flatten axis 0/1/-1 + int64 + bool exact | pass 4/4 new | 2026-10-18 | RTX 3060; bytes preserved on every dtype |
| V-F2 | Flatten bad axis negative | pass | 2026-10-18 | Names node + op |
| V-G1 | Gemm 2x2 hand-computed + oracle | pass | 2026-10-18 | {19,22,43,50} |
| V-G2 | Gemm transA/transB + alpha/beta + [N]/[M,N] bias | pass 2/2 new | 2026-10-18 | CPU-oracle compare; float accumulation, tolerance |
| V-G3 | Gemm negatives: contraction, bias, output | pass 3/3 | 2026-10-18 | Each names node + op |
| V-D1 | OFF leg full `ctest` | pass 12/12 | 2026-10-18 | Unchanged |
| V-D2 | CUDA-ON leg full `ctest` | pass 24/24 | 2026-10-18 | Unchanged |
| V-D3 | OpenCL-ON leg full `ctest` | pass 14/14 | 2026-10-18 | Table 18→20 rows |
| V-D4 | `scripts/format.sh --check` | pass | 2026-10-18 | Post-fix rerun |
| V-D5 | Diff audit | pass | 2026-10-18 | Packet paths only |

## Deviations

- `implementer` exhausted its step budget during research (no files
  written); the orchestrator implemented the packet directly.
- The OpenCL no-row probe moved from `Gemm` to `Conv` (Gemm now has a
  row; Conv remains the permanently missing portable op). Same for the
  device-table row assertions.
- Gemm accumulates in float32 while the CPU oracle uses double; all
  comparisons use `ExpectNearOcl`, and the reported values match.
