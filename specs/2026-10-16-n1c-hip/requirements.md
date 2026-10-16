# Requirements — N1c HIP Port (hipify)

Date: 2026-10-16. Parent packet: `specs/2026-10-16-n1-cuda/` (hybrid
vendor decision: AMD ROCm-supported discrete → HIP via hipify from the
CUDA code). Base: `main` post-N1b.

## Goal

The N1 CUDA sources (`src/cuda/`, `src/cuda_device.cpp`) build as HIP
for AMD ROCm-supported discrete GPUs, with the same oracle-parity bar.

## In Scope

- [R-1] `hipify-perl` translation of `src/cuda/*.cu` + `cuda_device.cpp`
  into a `src/hip/` tree (mechanical, checked-in translated sources, not
  a build-time hipify step — reviewable diffs).
- [R-2] Build wiring: `NERT_ENABLE_HIP=ON` (OFF default, mutually
  exclusive with `NERT_ENABLE_CUDA`), `hipcc` language, rocBLAS + MIOpen
  (the cuBLAS/cuDNN counterparts) link-PRIVATE.
- [R-3] `HipDevice` behind the same seam (`is_host() == false`);
  executor staging already covers it.
- [R-4] Parity: same suites as N1a/N1b (`nert_hip_*`) within
  `1e-3·(1+|ref|)` on an AMD runner.

## Out of Scope

- New ops; Vulkan/OpenCL portable backend (N1d); fusion (N1e); gfx-arch
  backports beyond what hipify handles.

## Status: BLOCKED (2026-10-16)

No AMD GPU, no ROCm/hipcc/hipify on this box or in `nert-gpu:26.08`
(NVIDIA-only). Unverified source translation will NOT be committed —
correctness rests on a compile + GPU run. Unblocks on an AMD runner
(RDNA2+ with ROCm) or a ROCm devcontainer on AMD hardware.

## Constraints (for the unblocking run)

- Same bars as N1a/N1b: OFF build untouched, format clean (`.hip` in
  `scripts/format.sh` scope if added), no `src/kernels`/loader/shape
  diffs, per-slice validation evidence before merge.
