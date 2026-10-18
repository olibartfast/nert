# OpenCL Movement Operators

Date: 2026-10-18.

## Purpose

CUDA covers every CPU operator used by the current models. The portable
backends are the remaining gap: OpenCL and HIP began with nine operators each,
and no full model can run on either until `Conv` and the movement operators are
available.

The coverage matrix in
[`specs/2026-10-17-task-models/coverage.md`](../2026-10-17-task-models/coverage.md)
ranks operators by model usage and node touches. Shared movement operators are
the best first investment because they are architecture-independent and reused
across detection, segmentation, and open-vocabulary graphs.

| Operator | Models | Node touches |
| --- | ---: | ---: |
| `Transpose` | 4 | 252 |
| `Slice` | 4 | 85 |
| `Concat` | 4 | 83 |
| `Shape` | 4 | 41 |
| `Unsqueeze` | 4 | 37 |
| `Gather` | 4 | 22 |

## Scope

Implement the six operators as first-party OpenCL C in `src/opencl/`, compiled
through `opencl_runtime`. Add table rows and target sources, and test each
operator in `test/opencl_kernels_test.cpp` with a hand-computed case, a CPU
oracle comparison, and negatives. Tests skip without an OpenCL GPU platform.
The rank cap of 16 on packed-index operators is the established pilot limit.

## Behavior

| Operator | Data types | Rules |
| --- | --- | --- |
| `Transpose` | float32, int64, bool | `perm` attribute, default reverse; gather each output through aligned input and output strides. |
| `Slice` | float32, int64, bool | Required int64 `starts`/`ends`, optional `axes`/`steps`; parameters downloaded to host; negative wrapping and clamping follow the oracle; zero-sized output is valid. |
| `Concat` | float32, int64, bool | Shared dtype and rank; negative axis wraps; contiguous device-to-device block copies; supports empty operands. |
| `Shape` | float32, int64, bool | int64 output; `start`/`end` attributes with negative wrapping and clamping. |
| `Unsqueeze` | float32, int64, bool | int64 `axes` input; wrap, sort, reject duplicates, and range-check; buffer copy, with shape owned by inference. |
| `Gather` | float32, int64, bool | int64 indices; `axis` defaults to 0 and wraps; negative indices wrap; out-of-range indices fail with node and operator named. |

## Boundaries

Out of scope: HIP mirrors (no hipify toolchain, recorded in `Q-1`), OpenCL
`Conv`, pools, bench fixtures, loader and shape-inference changes, executor
changes, and Neuriplo model runs.

The default build remains standard-library-only; OpenCL is private and opt-in.
A graph uses one device with no per-operator fallback. Kernels stay naive and
readable, and synchronize with `clFinish` following the existing `Relu` row.

## Decisions

The coverage matrix determined the six operators and their order; the
maintainer approved them on 2026-10-18. Pools follow, then `Conv`. No open
questions remain.
