# Plan — N1c HIP on NVIDIA

Worktree per milestone (`wip/nert-n1c<n>-<m>`), host RTX 3060 verifies
every slice through the HIP NVIDIA backend (`nvcc` + `hip-dev`
headers from scratch until a packaged install exists).

## M1 — scaffold (this slice)

- Root `CMakeLists.txt`: `option(NERT_ENABLE_HIP ...)`, CUDAToolkit
  required when ON (NVIDIA target), `find_path` for
  `hip/hip_runtime_api.h` (`HIP_INCLUDE_DIR` / `$HIP_PATH/include`),
  `src/hip_device.cpp` compiled as CUDA, `CUDA::cudart` link-PRIVATE.
- `test/CMakeLists.txt`: `nert_hip_device_test` under the flag.
- `test/hip_device_test.cpp` (same shape as the OpenCL M1 test;
  presence probe over `HipDevice()` construction, GTEST_SKIP in body).

## M2+ — kernels

- `hipify-perl` over the matching `src/cuda/` source into `src/hip/`,
  hand-reviewed (warp-size assumptions, `__shfl` variants, library
  calls never survive translation). Relu/Add first (mirror the CUDA
  and OpenCL pilots), then the first-party table. Parity vs CPU
  oracle within `1e-3·(1+|ref|)`; per-slice evidence in N1c-NVIDIA
  validation.
- **M6 (2026-10-22): 68-row first-party table.** All non-library CUDA
  kernels are hipified into `src/hip/`; the table excludes `Gemm`,
  `MatMul`, `Conv`, and `ConvTranspose`, which need rocBLAS/hipBLAS/
  MIOpen. The table probe and adapted CUDA first-party suites run on
  the RTX 3060 through the NVIDIA HIP backend.

## AMD half (still queued on the AMD runner)

- `HIP_PLATFORM=amd` build (hipcc/clang), rocBLAS `Gemm`,
  MIOpen `Conv`, full parity — per the 2026-10-16 N1c packet.

## Validation

Per-slice evidence lands in
`specs/2026-10-17-n1c-hip-nvidia/validation.md` — one table.
