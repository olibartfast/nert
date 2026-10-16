# Plan — N1c HIP Port (hipify)

Blocked on an AMD runner (see requirements Status). When unblocked,
branch `wip/nert-n1c` from main:

## M1 — Translate (AMD box or ROCm container on AMD hardware)

- Run `hipify-perl` over `src/cuda/*.cu` + `src/cuda_device.cpp` into
  `src/hip/`; hand-review every translated file (hipify misses:
  warp-size assumptions, `__shfl` variants, cub intrinsics if any land
  by then, cuBLAS→rocBLAS / cuDNN→MIOpen call mapping).
- rocBLAS `rocblas_ssgemm` replaces `cublasSgemm` (same row-major swap
  trick); MIOpen convolution replaces the cuDNN path (front-end API
  differs — expect a rewrite of `conv`, not a 1:1 port).

## M2 — Wire + parity

- `NERT_ENABLE_HIP` build flag, `HipDevice`, `nert_hip_*` tests mirroring
  the CUDA suites, parity evidence on the AMD runner.

## Replan note (2026-10-16)

N1c waits on hardware; meanwhile N1 coverage grows verifiably on the
RTX 3060 via first-party CUDA batches (N1e slices: elementwise,
pooling, normalization), which hipify later consumes unchanged in
structure.
