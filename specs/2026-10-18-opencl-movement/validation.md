# Validation — OpenCL shared movement rows

Defined before implementation. Tolerance: `ExpectNear` bench rule
unless exact. SKIP without an OpenCL GPU platform is recorded.

## Checks

- [V-A] Transpose (swap/default/neg-wrap, int64) + Gather
  (axis/neg-index) CPU vs OpenCL; negatives (bad perm, bad axis).
- [V-B] Slice (neg-axis/stepped) + Concat (neg-axis/multi-input)
  CPU vs OpenCL; negatives (oob clamp behavior per oracle, axis).
- [V-C] Shape (start/end slice) + Unsqueeze (neg axes) CPU vs
  OpenCL; negatives (dup axes, bad axis).
- [V-D1] OFF leg full `ctest` green. [V-D2] CUDA-ON leg full `ctest`
  green. [V-D3] OpenCL-ON leg (`-DNERT_ENABLE_OPENCL=ON`) full
  `ctest` green (RTX 3060, NVIDIA platform). [V-D4] format clean.
  [V-D5] diff audit: packet paths only.

## Acceptance (once at the end)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel "$(nproc)" && ctest --test-dir build --output-on-failure && ./scripts/format.sh --check
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DNERT_ENABLE_CUDA=ON && cmake --build build-cuda --parallel "$(nproc)" && ctest --test-dir build-cuda --output-on-failure
cmake -S . -B build-ocl -DCMAKE_BUILD_TYPE=Release -DNERT_ENABLE_OPENCL=ON && cmake --build build-ocl --parallel "$(nproc)" && ctest --test-dir build-ocl --output-on-failure
```

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-A | `nert_opencl_kernels_test` Transpose×3 + Gather×2 | pass 5/5 new | 2026-10-18 | RTX 3060 NVIDIA platform; atomic OOB-flag path green |
| V-B | Slice + Concat ×4 | pass 4/4 new | 2026-10-18 | Test-axes slip fixed (oracle right); CudaDevice typo caught on re-read |
| V-B2 | Slice negative-axis/step + Concat empty operand + Concat bool | pass 3/3 new | 2026-10-18 | Review follow-up: claimed variants now covered; Concat also gained the bool row (CPU/CUDA parity) |
| V-C | Shape + Unsqueeze ×4 | pass 4/4 new | 2026-10-18 | H2D/D2D transfer rows, no compute kernels |
| V-D1 | OFF leg full `ctest` | pass 12/12 | 2026-10-18 | Fixture fallback |
| V-D2 | CUDA-ON leg full `ctest` | pass 24/24 | 2026-10-18 | Unchanged by this slice |
| V-D3 | OpenCL-ON leg full `ctest` | pass 14/14 | 2026-10-18 | RTX 3060; table 9→15 rows |
| V-D4 | `scripts/format.sh --check` | pass | 2026-10-18 | `--fix` on new tests only |
| V-D5 | Diff audit + coverage.md sync | pass | 2026-10-18 | Packet paths + coverage matrix recompute (milestones 5/6/15/18/27) |

## Deviations

- `implementer` dispatch unavailable (harness); planner implements directly.
- Scope redirected from approved OpenCL pools to shared movement
  per Hermes coverage prioritization (maintainer-approved replan).
- `coverage.md` (parallel session's living doc) updated for the six
  rows: baseline, per-model lists, priority table, milestones, order
  sentence. Their in-flight content edits (README/docs/memory
  packet) untouched.
