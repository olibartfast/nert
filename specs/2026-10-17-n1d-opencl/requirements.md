# Requirements — N1d OpenCL Portable Backend (M1 scaffold)
Date: 2026-10-17. Base: `main` post-N1g gate. Parent: N1 hybrid decision.

## Reordering note (verifiability-first)

The N1 decision said Vulkan-compute first, then OpenCL. On this box the
Vulkan path is NOT verifiable: loader + NVIDIA ICD exist, but there is no
GLSL→SPIR-V toolchain (no glslc/glslang/shaderc) and no sudo to install
one — vendoring opaque precompiled SPIR-V would violate reviewability.
OpenCL is fully present (headers, libOpenCL, NVIDIA CUDA platform,
runtime kernel compilation — no toolchain needed). So: **OpenCL first,
Vulkan second (needs toolchain), HIP blocked (needs AMD hardware)**.
Unverified code is not committed; maintainer may override.

## Goal (M1)

An `OpenCLDevice` behind the same seam (`is_host() == false`, executor
staging covers it) with allocator + transfers + an empty kernel table,
proven on the RTX 3060 via the NVIDIA CUDA OpenCL platform.

## In Scope (M1)

- [R-1] `NERT_ENABLE_OPENCL=ON` (OFF default, independent of
  `NERT_ENABLE_CUDA`); configure FATALs naming the flag when headers/lib
  are absent. Raw OpenCL C API (no wrapper dep), RAII locally.
- [R-2] `src/opencl_device.cpp`: first NVIDIA-platform GPU, context +
  command queue, `clCreateBuffer` allocator (non-positive → null),
  `host_to_device`/`device_to_host` transfers, empty kernel table.
- [R-3] `test/opencl_device_test.cpp`: names itself, empty table resolves
  null (executor rejects naming op + node — covered once in M2, noted),
  1 KiB transfer round trip. SKIP without an OpenCL GPU platform.

## Out of Scope

- Any OpenCL kernel (M2+); Vulkan (toolchain-blocked, queued); HIP
  (hardware-blocked); executor/loader/shape changes.

## Constraints

- Same bars: OFF untouched, format clean, no `src/kernels`/loader/shape/
  executor diffs, SKIP contract intact.
