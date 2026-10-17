# Validation — N1d OpenCL Portable Backend

Defined before implementation. Bars: OFF untouched, format clean, no
`src/kernels`/loader/shape/executor diffs.

## Checks

- [V-1] M1: device names itself, empty table, 1 KiB round trip on the
  NVIDIA platform; OFF 8/8; format clean.
- [V-2] M2+: per-op parity within `1e-3·(1+|ref|)` on the RTX 3060.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| M1 | OpenCLDevice names/seam/empty-table/1KiB round trip on NVIDIA CUDA platform + OFF 8/8 + OCL-leg 9/9 + format | pass | 2026-10-17 | host leg; NERT_ENABLE_OPENCL independent of CUDA flag. |
