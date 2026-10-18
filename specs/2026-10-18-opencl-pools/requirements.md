# Requirements — OpenCL pools

Date: 2026-10-18. Follow-up to the landed movement six
(`specs/2026-10-18-opencl-movement/`): `MaxPool`,
`AveragePool`, `GlobalAveragePool` on OpenCL, hand-written OpenCL C
(Q-1). Pools + Flatten + Gemm are the remaining ResNet-18 portable
rows; `Conv` stays the hard dependency for every model.

## In scope

- `src/opencl/max_pool.cpp`, `average_pool.cpp`,
  `global_average_pool.cpp`: first-party kernels ported from the
  verified CUDA rows (same validation, same math), runtime-compiled
  via `opencl_runtime`. Table rows, `target_sources`, tests in
  `test/opencl_kernels_test.cpp` (hand-computed + oracle-compare +
  negatives, `GTEST_SKIP` without a platform).
- Oracle policies: MaxPool (kernel_shape required, dilations,
  auto_pad NOTSET/VALID/SAME_*, ceil_mode, -inf OOB); AveragePool
  (dilations fixed {1,1} per oracle, count_include_pad with the
  ceil-past-input divisor rule); GlobalAveragePool (rank>=2,
  double accumulation, one rounding).

## Out of scope

- HIP mirrors (no hipify here), OpenCL `Conv`/`Flatten`/`Gemm`
  (own slices), `bench/`, loader/shape/executor, Neuriplo runs.

## Constraints

- stdlib-only default; OpenCL PRIVATE, opt-in. Rank cap 16 on
  packed params (pilot limit). `clFinish` sync (Relu precedent).

## Open questions

- None.
