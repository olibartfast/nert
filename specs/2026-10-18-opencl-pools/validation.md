# Validation — OpenCL Pooling Operators

Defined before implementation. Comparisons use the benchmark's `ExpectNear`
tolerance unless a check is exact. Tests record `SKIP` when no OpenCL GPU
platform is available.

## What Each Check Proves

**MaxPool.** A stride-2 case is checked against hand-computed values. A
dilated and padded case is compared with the CPU oracle. Negative cases cover
a missing `kernel_shape` and invalid padding. The checks exercise `ceil_mode`,
dilation, and negative infinity for out-of-bounds samples.

**AveragePool.** A stride-2 case is checked against hand-computed values. The
`count_include_pad` cases compare both settings with the CPU oracle, including
a window that `ceil_mode` extends past the input and the resulting divisor.
The negative case covers an invalid configuration.

**GlobalAveragePool.** A rank-4 input is checked against hand-computed channel
means. A rank-3 input covers the rank contract. A bad output shape is rejected
with a message naming the node and operator. The fp64 capability guard is
recorded so a device without double precision fails explicitly rather than
silently changing numeric behavior.

**Build matrix.** The default, CUDA, and OpenCL builds must pass their full
CTest suites. Formatting must be clean, and the diff must stay within the
expected paths.

## Acceptance Commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --parallel "$(nproc)" && \
    ctest --test-dir build --output-on-failure && \
    ./scripts/format.sh --check

cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
    -DNERT_ENABLE_CUDA=ON && \
    cmake --build build-cuda --parallel "$(nproc)" && \
    ctest --test-dir build-cuda --output-on-failure

cmake -S . -B build-ocl -DCMAKE_BUILD_TYPE=Release \
    -DNERT_ENABLE_OPENCL=ON && \
    cmake --build build-ocl --parallel "$(nproc)" && \
    ctest --test-dir build-ocl --output-on-failure
```

## Evidence

| ID | Check | Result | Date | Notes |
| --- | --- | --- | --- | --- |
| `V-P1` | `MaxPool`: stride-2, dilated and padded, `ceil_mode`, and negative cases | Pass, 4/4 new | 2026-10-18 | RTX 3060. Negative infinity for out-of-bounds, `ceil_mode`, and dilation pass. A bad-pads negative was added during review. |
| `V-P2` | `AveragePool`: stride-2, `count_include_pad` 0 and 1, ceil-past-input divisor, and a negative case | Pass, 4/4 new | 2026-10-18 | A test-dimension mistake and a 24-versus-16-byte download bug were fixed against the oracle. The ceil-plus-pad divisor case was added during review. |
| `V-P3` | `GlobalAveragePool`: rank-4, rank-3, and a negative case | Pass, 3/3 new | 2026-10-18 | Double accumulation matches the oracle bit-for-bit. The rank-3 case and the fp64-requirement guard were added during review. |
| `V-D1` | Default build full CTest | Pass, 12/12 | 2026-10-18 | Fixture fallback. |
| `V-D2` | CUDA build full CTest | Pass, 24/24 | 2026-10-18 | Unchanged by this work. |
| `V-D3` | OpenCL build full CTest | Pass, 14/14 | 2026-10-18 | RTX 3060. The table grew from 15 to 18 rows. |
| `V-D4` | `scripts/format.sh --check` | Pass | 2026-10-18 | Rebuilt and rerun after the review fixes. |
| `V-D5` | Diff audit and coverage-matrix synchronization | Pass | 2026-10-18 | Specification paths plus the matrix recomputation. |

## Deviations

- The implementer dispatch was unavailable in the harness, so the planner
  implemented the work directly.
