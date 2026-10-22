# Validation — N1c HIP Port (hipify)

Defined before implementation. Blocked on an AMD runner; nothing to
evidence yet.

## Checks

- [V-1] hipify translation committed as reviewable sources in `src/hip/`
  (no build-time translation step).
- [V-2] `NERT_ENABLE_HIP=ON` configures/builds with `hipcc` on the AMD
  runner; OFF and CUDA legs unaffected.
- [V-3] `nert_hip_*` parity within `1e-3·(1+|ref|)` on the AMD runner.
- [V-4] Scope: no `src/kernels`/loader/shape diffs; format clean.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| — | — | blocked | 2026-10-16 | no AMD GPU / ROCm / hipify here or in nert-gpu:26.08 |
| — | NVIDIA first-party translation (sibling `2026-10-17-n1c-hip-nvidia`) | pass | 2026-10-22 | 68 hipified rows validated on the RTX 3060 through the NVIDIA HIP backend; the AMD `HIP_PLATFORM=amd` half and rocBLAS/MIOpen rows remain blocked on AMD hardware. |
