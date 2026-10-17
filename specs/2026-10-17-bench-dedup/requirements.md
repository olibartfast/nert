# Requirements — Bench every kernel, test every kernel, duplicate nothing

Date: 2026-10-17. Maintainer directive: all shipped kernels get a unit
test and a bench row, with no duplicated code anywhere.

## Why this packet exists

The `bench/` folder grew two generations that overlap. The old harnesses
(`bench/gemm`, `bench/matmul`) time vendored kernel copies instead of
shipped code, and they were only kept because the OpenCL bench legs had
no runtime device before N1d. That reason is gone: the OpenCL device
landed (N1d-M1..M4) and the HIP device landed (N1c-M1..M3). The new
driver (`bench/runtime`) times shipped kernels through the public API,
but it only builds for CUDA. Meanwhile the OpenCL and HIP kernel tests
each carry their own copy of the same upload/download/shape helpers.

## In scope

- [R-1] Delete `bench/gemm`, `bench/matmul`, and any `bench/utils`
  helper used only by them. No vendored kernel copies remain.
- [R-2] Extend `bench/runtime` to all three devices (CUDA, OpenCL, HIP)
  through the public `nert/device.hpp` seam only. One driver, one
  fixture list per op — no per-backend copies of timing or oracle code.
- [R-3] Merge the duplicated test helpers. The OpenCL and HIP kernel
  tests each define their own upload, download, host/device view, and
  node helpers; extract one shared header both suites use (the way
  `test/cuda_test_helpers.hpp` serves the CUDA suites).
- [R-4] Coverage audit: every kernel row in the CUDA, OpenCL, and HIP
  tables has a unit test and a bench fixture. Gaps become roadmap rows,
  not silent holes.

## Out of scope

- New kernels (those stay in their N1c/N1d slices); Vulkan (still needs
  a shader toolchain); the AMD-backend half (still needs the AMD runner).
- Changing tolerance bars: exact for integer and bitwise-stable float
  elementwise ops, `1e-3·(1+|ref|)` for libm float ops, predicates for
  IEEE edge cases and div-by-zero throws.

## Constraints

- The default build stays unchanged (bench remains opt-in, OFF by
  default) and format stays clean.
- Bench devices resolve through the public seam; engine internals stay
  out of `bench/` ([D-2] from the N1f packet).
- Tests SKIP (never fail) when their device is absent.
