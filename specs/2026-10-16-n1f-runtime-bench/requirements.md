# Requirements — N1f Runtime-Backed Test/Bench Without Duplication
Date: 2026-10-16. Base: `main` post-N1e-3. Track: N1e first-party,
verifiable on the RTX 3060 host leg.

## Problem

- `test/cuda_*_test.cpp` (5 files) carry byte-identical helper blocks
  (`HasCudaDevice`, `HostFloat`, `Upload`, `Download`, `DeviceFloat`,
  `MakeNode`, `ExpectNear`) plus per-test int64 upload lambdas.
- `bench/gemm` + `bench/matmul` measure **vendored kernel copies**
  (`bench/gemm/cpu/gemm.cpp` duplicates the runtime oracle by design;
  `bench/gemm/cuda/gemm.cpp` is a tiled kernel while the runtime ships
  cuBLAS). Bench numbers never describe shipped code, and no bench covers
  any N1e op. The bench harness is not benchmarking anything we ship.

## Goal

One source of truth per kernel: `src/kernels` (oracle) + `src/cuda`
(device). Tests and bench drive both tables through the public device
seam (`nert/device.hpp`) — no kernel logic reimplemented anywhere.

## In Scope

- [R-1] Shared test helpers: new `test/cuda_test_helpers.hpp` with the
  common block (+ `HostInt64`/`DeviceInt64`/`UploadInt64`/`DownloadInt64`
  to absorb the per-test int64 lambdas); all 5 CUDA test files include
  it, keeping only their file-specific runners.
- [R-2] Runtime-backed bench driver `bench/runtime/` (`nert_bench_runtime`,
  ctest `bench_runtime`): links `nert`, builds one fixture per CUDA-table
  op (shapes + attrs + deterministic data), runs the CPU table timed,
  uploads once, runs the CUDA table timed (kernels self-synchronize, so
  `steady_clock` is honest end-to-end), compares within
  `1e-3·(1+|ref|)`, prints `op | shape | cpu_ms | gpu_ms | speedup |
  maxdiff`, exits non-zero on mismatch.
- [R-3] SKIP contract: no CUDA device → `SKIP:` line, CPU halves still
  run, exit 0; a missing CUDA row (e.g. Conv without cuDNN) skips that
  case, never fails. OFF builds and CPU-only runners stay green.
- [R-4] No CUDA headers in the driver: device presence is probed through
  the seam (`allocate(1)` null check), so the driver compiles wherever
  `nert` compiles.

## Out of Scope

- Deleting the legacy `bench/gemm` + `bench/matmul` harnesses: they still
  serve the OpenCL bench legs, which have no runtime device until N1d.
  Deletion is a follow-up tied to N1d, noted in `bench/README.md`.
- New ops; HIP/Vulkan; fusion.

## Constraints

- Same bars: OFF untouched, format clean, no `src/kernels`/loader/shape/
  executor diffs, per-slice validation evidence before merge.
- The `[D-2]` bench-independence rule is honored, not broken: the driver
  uses only public headers (`nert/device.hpp`), never engine internals.
