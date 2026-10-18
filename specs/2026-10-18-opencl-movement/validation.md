# Validation — OpenCL Movement Operators

Defined before implementation. Comparisons use the benchmark's `ExpectNear`
tolerance unless a check is exact. Tests record `SKIP` when no OpenCL GPU
platform is available.

## What Each Check Proves

**Transpose and Gather.** `Transpose` covers default and explicit permutations,
including a negative-wrap case and int64 data. `Gather` covers axis handling
and negative indices. Both are compared with the CPU oracle and include
negative cases for an invalid permutation and an invalid axis.

**Slice and Concat.** `Slice` covers negative axes, stepped slices, and
out-of-bounds clamping. `Concat` covers a negative axis and multiple inputs.
Both are compared with CPU. Negative cases cover invalid axes and the oracle's
out-of-bounds behavior. A follow-up check adds bool `Concat`, an empty
operand, and a negative-step `Slice`, which closes the variants that the first
pass did not exercise.

**Shape and Unsqueeze.** `Shape` covers `start`/`end` selection. `Unsqueeze`
covers negative axes. Both are compared with CPU; negative cases cover
duplicate axes and invalid axes.

**Build matrix.** The default build, the CUDA build, and the OpenCL build must
all pass their full CTest suites. Formatting must be clean, and the diff must
stay within the expected specification and source paths.

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
| `V-A` | `nert_opencl_kernels_test`: three `Transpose` cases and two `Gather` cases | Pass, 5/5 new | 2026-10-18 | RTX 3060 NVIDIA platform. The atomic out-of-bounds flag path passes. |
| `V-B` | `Slice` and `Concat`, four cases | Pass, 4/4 new | 2026-10-18 | A test-axis mistake was corrected against the oracle; a wrong-device typo was caught on re-read. |
| `V-B2` | `Slice` negative axis and negative step, `Concat` empty operand, and bool `Concat` | Pass, 3/3 new | 2026-10-18 | Review follow-up. The previously claimed variants are now covered, and `Concat` gained the bool row already present on CPU and CUDA. |
| `V-C` | `Shape` and `Unsqueeze`, four cases | Pass, 4/4 new | 2026-10-18 | Host-to-device and device-to-device transfer rows; no compute kernels. |
| `V-D1` | Default build full CTest | Pass, 12/12 | 2026-10-18 | Fixture fallback. |
| `V-D2` | CUDA build full CTest | Pass, 24/24 | 2026-10-18 | Unchanged by this work. |
| `V-D3` | OpenCL build full CTest | Pass, 14/14 | 2026-10-18 | RTX 3060. The table grew from 9 to 15 rows. |
| `V-D4` | `scripts/format.sh --check` | Pass | 2026-10-18 | `--fix` applied only to new tests. |
| `V-D5` | Diff audit and coverage-matrix synchronization | Pass | 2026-10-18 | Specification paths plus the coverage-matrix recomputation for milestones 5, 6, 15, 18, and 27. |

## Deviations

- The implementer dispatch was unavailable in the harness, so the planner
  implemented the work directly.
- The scope was redirected from the approved OpenCL pools to the shared
  movement operators after the coverage review. The maintainer approved the
  replan.
- The coverage matrix, owned by a parallel session, was updated for the six
  rows: baseline counts, per-model lists, priority table, milestones, and the
  ordering sentence. Its in-flight content edits to `README.md`, `docs/`, and
  the memory-footprint specification were left untouched.
