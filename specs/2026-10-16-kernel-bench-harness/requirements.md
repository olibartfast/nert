# Requirements — bench/ playground layout (GPU kernel testing & benchmarking)

Date: 2026-10-16. Branch: `feature/bench-playground-layout` (from `main`).
Constitution: `specs/mission.md`, `specs/tech-stack.md`, `specs/roadmap.md`.
Reference: `https://github.com/olibartfast/gpu-playground`
(`source/<kernel>/{main.cpp,cuda/,opencl/,opencl_cpp/}`, `source/utils/`,
`GPU_ENABLE_*` flags, `USE_OPENCL*` switch, `benchmark()` harness).
Prior answers carried over (2026-10-16): pilot Gemm, CUDA+OpenCL, vendored
utils, `.hpp`, modern C++ — relocated from Neuriplo (consumer) to NERT.

## Intent

Give NERT kernels a gpu-playground-style home for standalone testing and
benchmarking: per-kernel harnesses (CPU oracle vs device kernel + timings)
without touching the runtime's dependency-free, unoptimized CPU path.

## Scope

In:

- [R-1] Top-level `bench/` mirrors playground: `bench/utils/` +
  `bench/gemm/` pilot (`main.cpp`, `cuda/`, `opencl/`, `opencl_cpp/`,
  `CMakeLists.txt` each). `src/kernels/` runtime layout untouched.
- [R-2] Backend switch: exactly one of CUDA (default), OpenCL C API
  (`NERT_BENCH_USE_OPENCL=ON`), OpenCL C++ wrapper
  (`NERT_BENCH_USE_OPENCL_CPP=ON`) per configure. `main.cpp` is a 3-way
  `#ifdef` harness (small exact + large multi-seed tolerance tests,
  `benchmarkWithReset`, exit `0`/`1`, `SKIP:` + exit `0` for the device half
  when no GPU present so CPU-only CI stays green).
- [R-3] `bench/utils/` vendors playground timing/helpers with attribution:
  `benchmark.hpp`, `benchmark_helpers.hpp`, `cuda_helpers.hpp`,
  `opencl_c_helpers.hpp`, `opencl_helpers.hpp` + CMake INTERFACE targets
  (`nert_bench_cuda_helpers` etc.). **All headers `.hpp`.**
- [R-4] Pilot `gemm` (`C = α·A·B + β·C`, FP32): bench-local CPU oracle
  (double accumulation, one rounding — mirrors runtime [D-15], duplicated so
  the runtime boundary holds) + tiled CUDA kernel + OpenCL C/C++ ports with
  identical `gemmCpu`/`gemmGPU` signatures. `.cpp` compiled as CUDA
  (`LANGUAGE CUDA` property) keeps CMake 3.16 floor, as in playground.
- [R-5] Opt-in: `NERT_BUILD_BENCH=OFF` default; runtime link closure stays
  stdlib-only. CUDA/OpenCL SDKs required only when bench enabled for that
  backend. Default `cmake -S . -B build && cmake --build build && ctest`
  output unchanged when `OFF`.
- [R-6] TDD per feature: failing-then-passing check first, evidence in
  `validation.md` with dates.
- [R-7] Modern C++17, no C idioms: `std::vector` RAII (no `new[]`/`delete[]`,
  `malloc`/`free`); `<iostream>` (no `printf`); C++ casts; `constexpr`;
  helpers throw `std::runtime_error` with file/line (no `abort()` in library
  paths); no `using namespace std` in headers. Playground sources are the
  algorithm reference only — translate, don't paste.

Out:

- Other kernels (MatMul next, Tier batches later); no Python DSL, no
  `cuda_perf_analysis.sh` port.
- No `src/`, `include/`, `test/` changes; no device-layer wiring (Phase N1
  owns it); no namespace rename ([Q-5] untouched).

## Constraints

- [C-1] C++17, CMake ≥ 3.16 (no root `LANGUAGES CUDA`).
- [C-2] Runtime stays dependency-free: `nert` target gains no CUDA/OpenCL
  link; bench targets are separate and `OFF` by default.
- [C-3] `src/kernels/` oracle style untouched (naive, readable); bench CPU
  oracle duplicates rather than includes the private `Kernels.hpp`.
- [C-4] Mission boundary respected: `bench/` is tooling (like `tools/`),
  not runtime scope — "no CUDA in current scope" still holds for `src/`;
  N1 owns the device layer. This packet builds the measurement harness only.

## Decisions

- [D-1] `bench/` top-level (not `src/kernels/` subdirs): keeps the runtime
  one-file-per-op pattern and link closure intact. Harness matrix maps
  playground `GPU_ENABLE_*`/`USE_*` to `NERT_BUILD_BENCH*` names (table in
  `bench/README.md`) to avoid collisions when both trees coexist.
- [D-2] CPU oracle duplicated in `bench/gemm/cpu/` (not wired to
  `src/kernels/Gemm.cpp`): `Kernels.hpp` is private to the engine; bench
  must not include it.

## Open questions

- [Q-1] Vendor-kernel decision (first-party vs opt-in cuBLAS/cuDNN baseline):
  still required before N1 device-layer work; this packet assumes
  first-party tiled kernels for measurement only. Owner: maintainer.
