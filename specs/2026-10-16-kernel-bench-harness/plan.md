# Plan — bench/ playground layout

Branch: `feature/bench-playground-layout` (from `main`). Thin phases, each
ends runnable + reviewable. TDD red→green→refactor per phase. Autocompact:
summarize + clean scratch build dirs at each phase end.

## G1 — Scaffold + bench/utils (no kernels yet)

- Root `CMakeLists.txt` append: `option(NERT_BUILD_BENCH OFF)` +
  `NERT_BENCH_USE_OPENCL/OFF`, `NERT_BENCH_USE_OPENCL_CPP/OFF` (mutual
  exclusion `FATAL_ERROR` when both `ON`); `if(NERT_BUILD_BENCH)
  add_subdirectory(bench) endif()`. `bench/CMakeLists.txt` adds `utils/`.
- `bench/utils/`: `benchmark.hpp`, `benchmark_helpers.hpp`,
  `cuda_helpers.hpp`, `opencl_c_helpers.hpp`, `opencl_helpers.hpp`
  (translated per [R-7]) + `ATTRIBUTION.md` + `CMakeLists.txt` (INTERFACE
  targets `nert_bench_cuda_helpers` / `nert_bench_opencl_helpers` /
  `nert_bench_opencl_cpp_helpers`; CUDA uses `find_package(CUDAToolkit)`
  only inside bench so default configure never needs nvcc; OpenCL Cpp
  `OpenCL::HeadersCpp` optional).
- `bench/utils/test/headers_compile.cpp` as `ctest -R bench_utils_headers`:
  includes all five headers, asserts `summarize({1,2,3}).median_ms == 2.0`.
- TDD: OFF-configure shows no `bench_*` tests; ON-configure + test green.
- Files: `bench/{CMakeLists.txt,README.md}`, `bench/utils/*`, root hook.

## G2 — bench/gemm CPU oracle + harness (CPU-only green)

- `bench/gemm/{CMakeLists.txt,main.cpp,cpu/gemm.hpp,cpu/gemm.cpp}` +
  `cuda/gemm.hpp` fallback TU (`gemmGPU` throws `runtime_error("CUDA backend
  not enabled")` when CUDA not selected) so the harness links everywhere.
- Oracle: double accumulation, one rounding; `main.cpp` 3-way `#ifdef`,
  small 2×3×2 exact + large 512×256×512 ×5 seeds within `1e-3·(1+|ref|)`,
  `benchmarkWithReset`, `std::cout`, return `0`/`1`. Target `nert_bench_gemm`,
  test `bench_gemm` (CPU path, no GPU needed).
- TDD: `ctest -R bench_gemm` red before files, green after.

## G3 — CUDA backend (same harness measures GPU)

- `cuda/gemm.cpp` (`LANGUAGE CUDA`), tiled `gemmTiled` (TILE 16), RAII device
  buffer helper, throwing `CUDA_CHECK`; `gemmGPU` H2D/compute/D2H.
  `bench/gemm/CMakeLists.txt` selects `cuda/` when CUDAToolkit found and no
  OpenCL flag; `main.cpp` catches device-throw → `SKIP:` + exit `0` for the
  GPU half (CPU mismatch still fails).
- TDD: GPU runner compares + benchmarks; CPU-only runner passes via SKIP.

## G4 — OpenCL C + C++ backends

- `opencl/gemm.{hpp,cpp}` (C API, throwing `CL_CHECK`, tiled kernel string
  `-D TILE_SIZE=16`) + `opencl_cpp/gemm.{hpp,cpp}` (wrapper). CMake selects
  per flag; `main.cpp` include path already 3-way.
- TDD: each flag configures+builds+tests (SKIP without OpenCL GPU); both
  flags together fail at configure with a clear message.

## G5 — Docs, quality, close-out

- `bench/README.md` full (layout/flag map, commands, SKIP contract),
  roadmap N1 note (measurement harness exists; device layer still N1),
  `scripts/format.sh --check` clean, validation evidence filled, scratch
  build dirs removed.
- No `src/`, `include/`, `test/`, version changes.

Order: G1 → G2 → G3 → G4 → G5 (each depends on the previous). Reviewer pass
after G4, before G5 close-out.

## Decisions (build-time findings)

- [D-3] Bench `ON` requires both SDKs at configure (all helpers
  compile-checked together in one TU); which device runs stays per flag.
- [D-4] Bench floor is CMake >= 3.17 (CUDAToolkit module, explicit
  `FATAL_ERROR`); runtime floor stays 3.16.
- [D-5] G2 large-trial PASS means the oracle ran clean on 5 seeds (no device
  to compare yet); the small exact test is the G2 correctness proof, the
  device compare arrived in G3.
