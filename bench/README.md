# NERT bench driver

Standalone tooling for timing shipped kernels against the CPU oracle.
`bench/` lives outside the runtime: nothing here is linked into the
`nert` library, and the runtime stays stdlib-only.

`bench/runtime/main.cpp` implements zero kernel logic: one fixture per
op (shapes + attrs + fixed-seed data) drives the runtime's own device
tables, times both sides (`steady_clock` around the internally
synchronized calls), compares within `1e-3·(1+|ref|)`, and prints
`op | case | cpu_ms | gpu_ms | speedup | maxdiff`. New kernel coverage
means a new fixture — never a copied kernel.

## Flags

| flag | default | meaning |
| --- | --- | --- |
| `NERT_BUILD_BENCH` | `OFF` | build the bench driver at all |

Bench is opt-in precisely so the default build needs nothing beyond a
C++17 compiler. Which devices get benched is decided by the device
flags (`NERT_ENABLE_CUDA`, and later OpenCL/HIP): a device that is not
compiled in simply has no bench half, and a missing GPU at run time
prints `SKIP:` and still exits 0, so CPU-only runners stay green.

## Commands

```bash
cmake -S . -B build-cuda -DNERT_ENABLE_CUDA=ON -DNERT_BUILD_BENCH=ON
cmake --build build-cuda --target nert_bench_runtime
./build-cuda/bench/runtime/nert_bench_runtime
# or: ctest --test-dir build-cuda -R bench_runtime --output-on-failure
```

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

Softmax is only 2.9x: one thread per row with double accumulation on
both sides, honestly reported, not tuned.

## History

Before 2026-10-17 `bench/` also held per-kernel harnesses (`gemm`,
`matmul`) that timed vendored kernel copies instead of shipped code.
They were deleted once the OpenCL and HIP runtime devices landed, since
the reason to keep them (bench legs with no runtime device) was gone.
