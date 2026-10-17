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
| — | — | pending | — | — |
