# Plan — N1d OpenCL Portable Backend

Worktree per milestone (`wip/nert-n1d-<m>`), host RTX 3060 verifies every
slice on the NVIDIA CUDA OpenCL platform.

## M1 — scaffold (this slice)

- Root `CMakeLists.txt`: `option(NERT_ENABLE_OPENCL ...)`, `find_package
  (OpenCL)` guarded by the flag, `src/opencl_device.cpp` into `nert`,
  `OpenCL::OpenCL` link-PRIVATE.
- `test/CMakeLists.txt`: `nert_opencl_device_test` under the flag.
- `test/opencl_device_test.cpp` with the shared-helper spirit (own tiny
  presence probe over `clGetPlatformIDs`, never GTEST_SKIP in helpers).

## M2+ — kernels (queued, own slices)

- OpenCL C sources per op (runtime-compiled via `clBuildProgram`, build
  log on failure), starting with Relu/Add (mirror the CUDA pilot), then
  the YOLO set in N1g order. Parity vs CPU oracle within
  `1e-3·(1+|ref|)`; per-slice evidence in N1d validation.

## Validation

Per-slice evidence lands in
`specs/2026-10-17-n1d-opencl/validation.md` — one table.
