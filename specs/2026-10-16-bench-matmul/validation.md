# Validation — bench/matmul

Reconstructed after implementation from the base packet's contract
(procedural deviation noted in `plan.md`). Evidence below is measured,
2026-10-16, RTX 3060 Laptop GPU, CUDA 12.0, CMake 3.28.3.

## Checks

- [V-1] Harness: `./bench/matmul/nert_bench_matmul` prints exact small
  `[[58,64],[139,154]]`, exact batched broadcast (identity B), 5/5 large
  seeds, timings, exit `0` on all three backend legs.
- [V-2] Parity: CPU vs device within `1e-3·(1+|ref|)` (small/batched use the
  tighter `1e-4` exact bound and match exactly).
- [V-3] Matrix: default CUDA + `NERT_BENCH_USE_OPENCL=ON` +
  `NERT_BENCH_USE_OPENCL_CPP=ON` each `ctest -R bench_` 3/3 (headers, gemm
  regression, matmul).
- [V-4] Style: C-idiom grep over `bench/matmul/` empty;
  `clang-format-18 --dry-run --Werror` clean on all 9 matmul C++ files.
- [V-5] Scope: `git status` shows only `bench/matmul/`,
  `bench/CMakeLists.txt` (+1 line), `specs/2026-10-16-bench-matmul/`.

## Evidence

- CUDA leg: small exact + batched exact + 5/5 large PASSED, exit 0.
- OpenCL C leg: same, exit 0. OpenCL C++ leg: same, exit 0.
- `ctest -R bench_`: 3/3 on each leg (CUDA 8.4 s, OpenCL C 20.5 s, C++ 20.2 s).
- Format: clean. C-idiom grep: empty.
