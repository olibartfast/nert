# Plan — N1f Runtime-Backed Test/Bench

Single worktree `wip/nert-n1f`, host RTX 3060 verifies.

## M1 — Shared test helpers (no behavior change)

- New `test/cuda_test_helpers.hpp` (`#pragma once`, anonymous namespace,
  internal linkage per TU — exactly today's semantics): `HasCudaDevice`,
  `HostFloat`, `HostInt64`, `Upload`, `Download`, `UploadInt64`,
  `DownloadInt64`, `DeviceFloat`, `DeviceInt64`, `MakeNode`, `ExpectNear`.
- Migrate, in order: `cuda_device_test.cpp` (tiny block) →
  `cuda_pool_test.cpp` (keep `RunCudaPool`/`RunCpuPool`) →
  `cuda_activations_test.cpp` (keep `RunCuda`/`RunCpu`) →
  `cuda_conv_test.cpp` (keep `HasCudaConv` + conv runners) →
  `cuda_kernels_test.cpp` (also replace the 3 int64 lambda sets with the
  header helpers). Rebuild + full ON matrix after EACH file; stop on red.
- Header needs `<cuda_runtime.h>` (presence probe) + gtest (ExpectNear);
  CUDA test targets already satisfy both.

## M2 — `bench/runtime/` driver

- `bench/runtime/CMakeLists.txt`: `nert_bench_runtime` links `nert`
  (PRIVATE), `cxx_std_17`, `-Wall -Wextra -Wpedantic`,
  `add_test(NAME bench_runtime ...)` with `TIMEOUT 1200` (legacy parity);
  wired via `add_subdirectory(runtime)` in `bench/CMakeLists.txt`.
- `bench/runtime/main.cpp`: fixture = `{op, node+attrs, inputs (dims,
  dtype, fill), out dims, cpu iters, gpu iters}`. Deterministic fills
  (`mt19937` fixed seed; Div range excludes 0). Cases: Gemm 512³,
  Relu/Add/Mul/Sub/Div 4M (Add bcast `[2048,2048]+[2048]`),
  MaxPool/AveragePool `[8,64,112,112]` k3 s2, Sigmoid/Tanh 4M, Softmax
  `[256,4096]`, Conv `[8,16,64,64]` k3 + bias iff the CUDA row exists.
- CPU iters adaptive (1 iter if > 1.5 s, else 3); GPU warmup 3 + 10 timed.
  Mean ms reported; speedup = cpu/gpu; `maxdiff` + mismatch count.
- Exit 0 + `PASSED` iff every executed case matches; `SKIP:` lines for a
  missing device / missing row.

## Validation

Evidence lands in `specs/2026-10-16-n1f-runtime-bench/validation.md`:
ON (`-DNERT_ENABLE_CUDA=ON -DNERT_BUILD_BENCH=ON`) full ctest incl. the
new `bench_runtime` numbers table, OFF 8/8, format clean.
