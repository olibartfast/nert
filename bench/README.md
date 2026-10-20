# NERT bench driver

Standalone tooling for timing shipped kernels against the CPU oracle.
`bench/` lives outside the runtime: nothing here is linked into the
`nert` library, and the runtime stays stdlib-only.

`bench/runtime/main.cpp` implements zero kernel logic: one fixture per
op (shapes + attrs + fixed-seed data) drives the runtime's own device
tables for every device layer the binary was built with (CUDA, OpenCL,
HIP), times both sides (`steady_clock` around the internally
synchronized calls), compares within `1e-3·(1+|ref|)`, and prints
`op | case | device | cpu_ms | dev_ms | speedup | maxdiff`. New kernel
coverage means a new fixture — never a copied kernel.

## Flags

| flag | default | meaning |
| --- | --- | --- |
| `NERT_BUILD_BENCH` | `OFF` | build the bench driver at all |

Bench is opt-in precisely so the default build needs nothing beyond a
C++17 compiler. Which devices get benched is decided by the device
flags (`NERT_ENABLE_CUDA`, `NERT_ENABLE_OPENCL`, `NERT_ENABLE_HIP`):
one binary enumerates exactly the devices it was built with. A device
that is not compiled in has no bench half at all; one that is compiled
in but missing at run time prints `SKIP:` and still exits 0, so
CPU-only runners stay green. Device rows an op lacks also `SKIP:`, so
the same fixture list runs on every leg.

## Commands

```bash
# One leg per device layer; build any combination into one binary.
cmake -S . -B build-cuda -DNERT_ENABLE_CUDA=ON -DNERT_BUILD_BENCH=ON
cmake -S . -B build-ocl -DNERT_ENABLE_OPENCL=ON -DNERT_BUILD_BENCH=ON
cmake -S . -B build-hip -DNERT_ENABLE_HIP=ON -DNERT_BUILD_BENCH=ON \
    -DHIP_INCLUDE_DIR=/path/to/hip/include
cmake --build build-cuda --target nert_bench_runtime
./build-cuda/bench/runtime/nert_bench_runtime
# or: ctest --test-dir build-cuda -R bench_runtime --output-on-failure
```

Reference numbers (RTX 3060, 2026-10-17/18, one line per device leg):

| device | op | case | cpu_ms | dev_ms | speedup | maxdiff |
|--------|----|------|--------|--------|---------|---------|
| CUDA | Gemm | 512³ | 1637.50 | 0.65 | 2525.8x | 1.05e-05 |
| CUDA | Relu | 4M | 95.65 | 4.04 | 23.7x | 0 |
| CUDA | Sigmoid | 4M | 64.98 | 3.89 | 16.7x | 1.19e-07 |
| CUDA | Tanh | 4M | 252.57 | 4.14 | 61.1x | 1.19e-07 |
| CUDA | Add | 4M bcast | 212.01 | 9.69 | 21.9x | 0 |
| CUDA | Mul | 4M bcast | 192.97 | 9.63 | 20.0x | 0 |
| CUDA | Sub | 4M bcast | 211.43 | 9.68 | 21.8x | 0 |
| CUDA | Div | 4M | 117.45 | 9.47 | 12.4x | 0 |
| CUDA | MaxPool | [8,64,112,112] k3 s2 | 379.40 | 2.04 | 185.7x | 0 |
| CUDA | AveragePool | [8,64,112,112] k3 s2 | 137.04 | 1.47 | 93.2x | 0 |
| CUDA | Softmax | [256,4096] | 56.22 | 18.06 | 3.1x | 0 |
| CUDA | Conv | [8,16,64,64] k3+bias | SKIP (no cuDNN on host leg) | | | |
| OpenCL | Relu | 4M | 94.28 | 3.76 | 25.1x | 0 |
| OpenCL | Sigmoid | 4M | 63.53 | 3.78 | 16.8x | 1.79e-07 |
| OpenCL | Tanh | 4M | 244.65 | 3.77 | 64.9x | 1.19e-07 |
| OpenCL | Add | 4M bcast | 195.18 | 3.76 | 51.9x | 0 |
| OpenCL | Mul | 4M bcast | 203.69 | 3.82 | 53.3x | 0 |
| OpenCL | Sub | 4M bcast | 206.20 | 3.72 | 55.5x | 0 |
| OpenCL | Div | 4M | 112.31 | 4.03 | 27.9x | 4.77e-07 |
| OpenCL | Softmax | [256,4096] | 54.96 | 4.24 | 13.0x | 4.54e-09 |
| HIP | Relu | 4M | 91.12 | 4.19 | 21.8x | 0 |
| HIP | Sigmoid | 4M | 67.46 | 4.07 | 16.6x | 1.19e-07 |
| HIP | Tanh | 4M | 247.50 | 4.28 | 57.8x | 1.19e-07 |
| HIP | Add | 4M bcast | 196.64 | 9.57 | 20.6x | 0 |
| HIP | Mul | 4M bcast | 192.22 | 9.64 | 19.9x | 0 |
| HIP | Sub | 4M bcast | 188.75 | 9.65 | 19.6x | 0 |
| HIP | Div | 4M | 105.87 | 9.56 | 11.1x | 0 |
| HIP | Softmax | [256,4096] | 54.96 | 17.32 | 3.2x | 0 |
| CUDA | Reshape | [2,3,4] -> [6,4] | 0.00 | 0.03 | 0.1x | 0 |
| OpenCL | Reshape | [2,3,4] -> [6,4] | 0.00 | 0.09 | 0.0x | 0 |
| HIP | Reshape | [2,3,4] -> [6,4] | 0.00 | 0.02 | 0.1x | 0 |

### R6 / Tier-5 coverage (2026-10-20, RTX 3060)

Fixtures for the 31 R6 + Tier-5 CUDA rows and the norm corridor. These
are coverage rows, not tuning claims: `maxdiff` is absolute for float32
and 0/1 (exact) for int64/bool rows. OpenCL carries most of these rows
now; HIP still has only the original pilot set.

| op | case | CUDA ms / maxdiff | OpenCL ms / maxdiff |
|---|---|---|---|
| Sin | [1048576] | 0.92 / 5.96e-08 | 0.95 / 5.96e-08 |
| Cos | [1048576] | 0.91 / 5.96e-08 | 0.98 / 5.96e-08 |
| Exp | [1048576] | 0.91 / 1.91e-06 | 0.98 / 1.91e-06 |
| Elu | [1048576] | 0.92 / 9.69e-08 | 0.95 / 9.69e-08 |
| Log | [1048576] | 0.93 / 2.38e-07 | 0.93 / 2.38e-07 |
| Floor | [1048576] | 0.90 / 0.00e+00 | SKIP (no row) |
| Abs | [1048576] | 0.92 / 0.00e+00 | SKIP (no row) |
| Neg | [1048576] | 0.93 / 0.00e+00 | SKIP (no row) |
| Sqrt | [1048576] | 0.91 / 0.00e+00 | 0.94 / 2.38e-07 |
| Erf | [1048576] | 0.90 / 5.96e-08 | 0.97 / 5.96e-08 |
| IsNaN | [1048576] | 0.31 / 0.00e+00 | 0.33 / 0.00e+00 |
| Greater | [1024x1024] vs [1024] | 1.75 / 0.00e+00 | 0.34 / 0.00e+00 |
| Less | [1024x1024] vs [1024] | 1.75 / 0.00e+00 | 0.33 / 0.00e+00 |
| GreaterOrEqual | [1024x1024] vs [1024] | 1.74 / 0.00e+00 | SKIP (no row) |
| LessOrEqual | [1024x1024] vs [1024] | 1.75 / 0.00e+00 | 0.34 / 0.00e+00 |
| And | [1024x1024] bcast | 1.74 / 0.00e+00 | 0.34 / 0.00e+00 |
| Or | [1024x1024] bcast | 1.74 / 0.00e+00 | 0.39 / 0.00e+00 |
| Not | [1048576] | 0.30 / 0.00e+00 | 0.31 / 0.00e+00 |
| Pow | [1024x1024]^[1024] | 2.35 / 2.38e-07 | 0.98 / 2.38e-07 |
| Max | [1024x1024] fold3 | 3.84 / 0.00e+00 | 1.02 / 0.00e+00 |
| Min | [1024x1024] fold3 | 3.84 / 0.00e+00 | 1.03 / 0.00e+00 |
| Clip | [1048576] [0,1] | 0.92 / 0.00e+00 | 0.97 / 0.00e+00 |
| Identity | [1048576] | 0.89 / 0.00e+00 | 0.93 / 0.00e+00 |
| Squeeze | [1x1024x1] -> [1024] | 0.02 / 0.00e+00 | 0.03 / 0.00e+00 |
| Tile | [1024x1] x{2,3} | 0.04 / 0.00e+00 | 0.05 / 0.00e+00 |
| Range | 0..1M step 1 | 0.93 / 0.00e+00 | 0.97 / 0.00e+00 |
| EyeLike | [1024x1024] | 0.95 / 0.00e+00 | 1.39 / 0.00e+00 |
| Pad | [1024x1024] +1 | 2.93 / 0.00e+00 | 1.00 / 0.00e+00 |
| ReduceSum | [1024x1024] axis1 | 0.74 / 0.00e+00 | 0.47 / 0.00e+00 |
| ReduceMean | [1024x1024] axis1 | 0.75 / 0.00e+00 | 0.49 / 0.00e+00 |
| ReduceL2 | [1024x1024] axis1 | 0.75 / 0.00e+00 | 0.39 / 4.58e-05 |
| ArgMax | [1024x1024] axis1 | 0.21 / 0.00e+00 | 0.23 / 0.00e+00 |
| CumSum | [1024x1024] axis1 | 1.38 / 0.00e+00 | 14.79 / 0.00e+00 |
| ScatterND | [1024x1024] rows | 7.72 / 0.00e+00 | 7.37 / 0.00e+00 |
| LayerNormalization | [1024x1024] axis1 | 2.13 / 0.00e+00 | 1.58 / 1.31e-06 |
| GlobalAveragePool | [8x64x112x112] | 2.75 / 0.00e+00 | 2.01 / 0.00e+00 |

Rows without a device kernel `SKIP` on that leg: CUDA has no cuDNN in
the host container (Conv), HIP has no Gemm/pools/Conv, and OpenCL has
no Floor/Abs/Neg/GreaterOrEqual. The driver supports mixed dtypes: a
fixture can carry float32, int64 (shape/axes/repeats) and bool tensors,
and int64/bool outputs compare exactly (maxdiff 0 or 1). The Reshape
fixture's CPU side is a memcpy, so its ratio is noise.
Softmax is only ~3x on CUDA/HIP: one thread per row with double
accumulation on both sides, honestly reported, not tuned.

## History

Before 2026-10-17 `bench/` also held per-kernel harnesses (`gemm`,
`matmul`) that timed vendored kernel copies instead of shipped code.
They were deleted once the OpenCL and HIP runtime devices landed, since
the reason to keep them (bench legs with no runtime device) was gone.

2026-10-18 (Slice 2): the driver enumerates every enabled device layer
behind one fixture list and prints a `device` column; the legacy
CUDA-only `gpu_ms` column became `dev_ms`. Building with no device
flags produces no bench target.

2026-10-20 (Slice 5b): typed fixtures (float32/int64/bool, mixed per
case) cover the R6/Tier-5 CUDA rows and the norm corridor on every
device that has them.
