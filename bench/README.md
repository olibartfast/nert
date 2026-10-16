# NERT bench harness

Standalone tooling for testing and timing individual kernels against a
CPU oracle and a device backend. `bench/` mirrors the layout of the
[gpu-playground](https://github.com/olibartfast/gpu-playground) reference but
lives outside the runtime: nothing here is linked into the `nert` library, and
the runtime stays stdlib-only ([C-2]).

This is a stub; the full layout/usage guide lands with G5.

## Layout (playground → nert)

| playground | nert | notes |
| --- | --- | --- |
| `source/utils/` | `bench/utils/` | vendored timing/helpers, all headers `.hpp` |
| `source/<kernel>/` | `bench/<kernel>/` | per-kernel harness (`main.cpp`, `cuda/`, `opencl/`, `opencl_cpp/`) |
| `GPU_ENABLE_*` | `NERT_BENCH_USE_*` | per-backend configure switch |
| `USE_OPENCL*` | `NERT_BENCH_USE_OPENCL*` | OpenCL C / C++ backend selection |

## Flags

| flag | default | meaning |
| --- | --- | --- |
| `NERT_BUILD_BENCH` | `OFF` | build the bench harness at all |
| `NERT_BENCH_USE_OPENCL` | `OFF` | OpenCL C API backend (mutually exclusive with the C++ one) |
| `NERT_BENCH_USE_OPENCL_CPP` | `OFF` | OpenCL C++ wrapper backend |

With `NERT_BUILD_BENCH=ON` and neither OpenCL flag set, the CUDA backend is
selected. Enabling both OpenCL flags is a configure-time `FATAL_ERROR`.

## Commands

```bash
cmake -S . -B build                       # default: no bench, runtime unchanged
cmake -S . -B build -DNERT_BUILD_BENCH=ON # enable the harness
cmake --build build
ctest --test-dir build -R bench_ --output-on-failure
```

## Kernels: gemm pilot

```bash
# CUDA path (default when bench is ON)
cmake -S . -B build-cuda -DNERT_BUILD_BENCH=ON && cmake --build build-cuda --target nert_bench_gemm
./build-cuda/bench/gemm/nert_bench_gemm

# OpenCL C API
cmake -S . -B build-ocl -DNERT_BUILD_BENCH=ON -DNERT_BENCH_USE_OPENCL=ON && cmake --build build-ocl --target nert_bench_gemm
./build-ocl/bench/gemm/nert_bench_gemm

# OpenCL C++ wrapper
cmake -S . -B build-oclcpp -DNERT_BUILD_BENCH=ON -DNERT_BENCH_USE_OPENCL_CPP=ON && cmake --build build-oclcpp --target nert_bench_gemm
./build-oclcpp/bench/gemm/nert_bench_gemm
```

Expected on a GPU runner (RTX 3060, 2026-10-16): small test prints
`[[58,64],[139,154]]`, all 5 large seeds PASSED with max abs diff `0.00`,
exit `0`. CUDA large is ~300x the naive CPU oracle; OpenCL legs ~2.3x
(they rebuild + transfer per call — honestly reported, not tuned).

## SDK prerequisites

Bench is opt-in precisely so the default build needs nothing beyond a C++17
compiler: with `NERT_BUILD_BENCH=OFF` (the default) no CUDA or OpenCL SDK is
touched and the CMake 3.16 floor holds. With `NERT_BUILD_BENCH=ON`, both the CUDA toolkit and the OpenCL SDK
are required at configure time (all helper headers are compile-checked
together via `bench_utils_headers`); a missing SDK fails naming the bench
flag. Bench also needs CMake >= 3.17 (CUDAToolkit module; explicit
`FATAL_ERROR` otherwise) while the runtime keeps its 3.16 floor. Which device actually runs is still chosen by the backend flags below,
and every device half runtime-SKIPs without a GPU, so CPU-only runners with
the SDKs installed stay green.

## `SKIP:` contract

Device halves exit `0` and print a `SKIP:` line when no usable GPU (or SDK) is
present, so CPU-only CI stays green; the CPU oracle half still runs and still
fails on a genuine mismatch.

[C-2]: ../specs/2026-10-16-kernel-bench-harness/requirements.md
