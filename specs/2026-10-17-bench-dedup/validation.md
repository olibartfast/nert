# Validation — Bench every kernel, test every kernel, duplicate nothing

Defined before implementation. Bars: default OFF build untouched,
format clean, bench stays opt-in and out of the `nert` link closure.

## Checks

- [V-1] Slice 1: no `bench/gemm`, `bench/matmul`, or `bench/utils`
  remnant; OFF 8/8; CUDA bench leg builds and runs; format clean.
- [V-2] Slice 2: bench fixtures run per enabled device; numbers in
  `bench/README.md`.
- [V-3] Slice 3: shared helper header in use; legs green; line count down.
- [V-4] Slice 4: per-row unit-test + bench-fixture checklist complete.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| Slice 1 | `bench/gemm,matmul,utils` deleted + flags dropped + README rewritten; OFF 8/8 + CUDA bench 11 pass/1 SKIP (Conv, no cuDNN) + format | pass | 2026-10-17 | `bench/` is now `runtime` only; numbers match README magnitude. |
| Slice 2 | One driver, one fixture list, `device` column; CUDA 11 pass/1 SKIP, OpenCL 8 pass/4 SKIP, HIP 8 pass/4 SKIP; OFF 12/12 + bench-only configure (no device) builds nothing + format | pass | 2026-10-18 | CUDA/OCL/HIP numbers in `bench/README.md`; missing row/device SKIPs, never fails. |
| Slice 3 | `test/device_test_helpers.hpp` shared by both GPU kernel suites + both device suites; OpenCL 2/2, HIP 2/2, OFF 12/12, format | pass | 2026-10-18 | 4 files -142 lines, +102-line header = net -40; matrix unchanged. |
| Slice 4 | Table walk (CPU 72, CUDA 68 at audit time / 69 after the OWLv2 CUDA Einsum merge, OpenCL 9, HIP 9 rows): every device row has a unit test; bench fixtures cover 12 ops (Gemm/Conv/pools/elementwise/Softmax). Gaps filed: R6/Tier-5 bench fixtures + Reshape fixture (OpenCL/HIP/CUDA/CPU) | pass | 2026-10-18 | Remaining legacy rows are coverage-only, not regressions; grouped into one follow-up slice rather than 56 entries. |
| Slice 5a | Reshape fixture with an int64 shape operand (driver now supports mixed int64 inputs): CUDA/OpenCL/HIP all PASS, maxdiff 0; CUDA 12 pass/1 skip, OCL 9/4, HIP 9/4 | pass | 2026-10-18 | Closes the only portable-backend row without a fixture; sub-ms CPU side makes ratios noise. |
| — | — | pending | — | — |
