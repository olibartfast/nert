# Validation — OpenCL Conv

Defined before implementation. Tolerance: `ExpectNearOcl` unless exact;
`GTEST_SKIP` without an OpenCL GPU platform is recorded.

## Checks

- [V-C1] 2-D Conv hand-computed (3x3 * 2x2) + oracle.
- [V-C2] Grouped/depthwise Conv (group=2) exact.
- [V-C3] Stride 2 + pads 1 + bias vs the CPU oracle.
- [V-C4] 1-D Conv rank lift exact.
- [V-C5] Negatives: weight-channel mismatch, bad bias, wrong output.
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
| V-C1 | Conv 2-D 3x3 * 2x2 hand-computed | pass | 2026-10-18 | {6,8,12,14}; oracle match |
| V-C2 | Grouped Conv (group=2, 1x1) | pass | 2026-10-18 | Per-channel scaling exact |
| V-C3 | Stride 2 + pads 1 + bias | pass | 2026-10-18 | 3x3 output, oracle compare |
| V-C4 | 1-D Conv rank lift | pass | 2026-10-18 | {3,5,7} |
| V-C5 | Conv negatives (channels/bias/output) | pass 3/3 | 2026-10-18 | Each names node + op |
| V-D1 | OFF leg full `ctest` | pass 12/12 | 2026-10-18 | Unchanged |
| V-D2 | CUDA-ON leg full `ctest` | pass 24/24 | 2026-10-18 | Unchanged |
| V-D3 | OpenCL-ON leg full `ctest` | pass 14/14 | 2026-10-18 | Table 20→21 rows |
| V-D4 | `scripts/format.sh --check` | pass | 2026-10-18 | |
| V-D5 | Diff audit | pass | 2026-10-18 | Packet paths only |

## Deviations

- The no-row probes move from `Conv` to `Sin` (Conv now has a row).
- Float accumulation vs the oracle's double; tolerance-compared.
