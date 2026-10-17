# Requirements — N1c HIP on NVIDIA (M1 scaffold)

Date: 2026-10-17. Base: `main` post-N1d-M3. Parent: N1 hybrid decision
(HIP via hipify on AMD ROCm) + `specs/2026-10-16-n1c-hip/` (AMD-runner
plan, still valid for AMD hardware).

## Unblock note (this box CAN verify HIP)

HIP supports NVIDIA GPUs: with `HIP_PLATFORM=nvidia` (i.e. defining
`__HIP_PLATFORM_NVIDIA__`) `hipcc` just drives `nvcc` and the HIP
runtime headers map `hipMalloc`/`hipMemcpy`/etc. onto CUDA. Proven here
on 2026-10-17, all without sudo or AMD hardware:

- `hip-dev_6.2` .deb fetched from AMD's apt pool (`repo.radeon.com`)
  and extracted rootless with `dpkg-deb -x` to scratch
  (`/tmp/opencode/rocm/` — never committed); it carries
  `include/hip/nvidia_detail/` (absent from the GitHub snapshot).
- A `hipMalloc`/`hipFree` probe compiled with host `nvcc` 12.0
  (`-D__HIP_PLATFORM_NVIDIA__ -I<hip>/include`, link `cudart`) and RAN
  on the RTX 3060, exit 0.
- `hipify-perl` (cloned `ROCm/HIPIFY`, script at `bin/hipify-perl`)
  translates our CUDA sources: `src/cuda/relu.cu` → `hipGetLastError`,
  `hipDeviceSynchronize`, etc. (scratch proof in `/tmp`).

So N1c proceeds on the RTX 3060 in the HIP dialect; the AMD runner
still owns the `HIP_PLATFORM=amd` (rocBLAS/MIOpen) half later.

## Goal (M1)

A `HipDevice` behind the same seam (`is_host() == false`, executor
staging covers it) with allocator + transfers + an empty kernel table,
proven on the RTX 3060 via the HIP NVIDIA backend.

## In Scope (M1)

- [R-1] `NERT_ENABLE_HIP=ON` (OFF default, independent of
  `NERT_ENABLE_CUDA`/`NERT_ENABLE_OPENCL`). ON requires CUDAToolkit
  (NVIDIA target drives nvcc) + HIP headers (`HIP_INCLUDE_DIR` cache
  var or `$ENV{HIP_PATH}/include`); configure FATALs naming the flag
  when either is absent. Raw HIP runtime API only.
- [R-2] `src/hip_device.cpp` (compiled as CUDA for the NVIDIA target):
  `hipMalloc`/`hipFree` allocator (non-positive → null),
  `hipMemcpy` transfers, empty kernel table.
- [R-3] `test/hip_device_test.cpp`: names itself, empty table resolves
  null (executor rejects naming op + node — covered once in M2, noted),
  1 KiB transfer round trip. SKIP without a HIP-usable GPU.

## Out of Scope

- Any HIP kernel (M2+); the AMD-backend half (rocBLAS/MIOpen, needs the
  AMD runner); executor/loader/shape changes.

## Constraints

- Same bars: OFF untouched, format clean, no `src/kernels`/loader/
  shape/executor diffs, SKIP contract intact.
- Scratch toolchain (`/tmp/opencode/rocm/`, `/tmp/opencode/hipify/`)
  is never committed; CMake consumes it via `HIP_INCLUDE_DIR`.
