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

## Kernels: gemm pilot (vendored copies)

The `gemm` / `matmul` harnesses below measure **vendored kernel copies**,
not shipped code (`cpu/gemm.cpp` duplicates the runtime oracle by design;
`cuda/gemm.cpp` is a tiled kernel while the runtime ships cuBLAS). They
remain the algorithm playground and still serve the OpenCL bench legs,
which have no runtime device until N1d — but they are NOT runtime
benchmarks. For shipped-kernel numbers use `bench/runtime` (next
section); deleting the vendored CUDA copies is a follow-up tied to N1d.

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

## Runtime bench: shipped kernels, no duplicated code

```bash
cmake -S . -B build-cuda -DNERT_ENABLE_CUDA=ON -DNERT_BUILD_BENCH=ON
cmake --build build-cuda --target nert_bench_runtime
./build-cuda/bench/runtime/nert_bench_runtime
# or: ctest --test-dir build-cuda -R bench_runtime --output-on-failure
```

`bench/runtime/main.cpp` implements zero kernel logic: one fixture per op
(shapes + attrs + fixed-seed data) drives the runtime's own
`CpuDevice`/`CudaDevice` tables, times both (`steady_clock` around the
internally synchronized calls), compares within `1e-3·(1+|ref|)`, and
prints `op | case | cpu_ms | gpu_ms | speedup | maxdiff`. New kernel
coverage = new fixture. Built only with `NERT_ENABLE_CUDA=ON` (OFF builds
carry no `CudaDevice` symbol); without a GPU, or for a missing CUDA row
(e.g. Conv without cuDNN), the case prints `SKIP:` and still exits 0.

Reference numbers (RTX 3060, 2026-10-16, 11 passed / 1 skipped):

| op | case | cpu_ms | gpu_ms | speedup | maxdiff |
|----|------|--------|--------|---------|---------|
| Gemm | 512³ | 1480.40 | 0.66 | 2227.7x | 1.05e-05 |
| Relu | 4M | 96.90 | 3.94 | 24.6x | 0 |
| Sigmoid | 4M | 63.80 | 3.88 | 16.4x | 1.19e-07 |
| Tanh | 4M | 248.98 | 4.17 | 59.7x | 1.19e-07 |
| Add | 4M bcast | 200.72 | 9.56 | 21.0x | 0 |
| Mul | 4M bcast | 198.88 | 9.62 | 20.7x | 0 |
| Sub | 4M bcast | 200.52 | 9.73 | 20.6x | 0 |
| Div | 4M | 116.31 | 9.57 | 12.2x | 0 |
| MaxPool | [8,64,112,112] k3 s2 | 382.15 | 1.76 | 217.6x | 0 |
| AveragePool | [8,64,112,112] k3 s2 | 146.76 | 1.64 | 89.6x | 0 |
| Softmax | [256,4096] | 58.92 | 20.22 | 2.9x | 0 |
| Conv | — | SKIP (no CUDA row: no cuDNN on this leg) | | | |

Softmax is only 2.9x: one thread per row with double accumulation on both
sides, honestly reported, not tuned.

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
