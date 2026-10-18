# Plan — OpenCL Pooling Operators

Work in the main tree. Do not modify `bench/`,
`specs/2026-10-17-cudnn-frontend-spike/`,
`specs/2026-10-18-memory-footprint/`, `docs/`, or `.github/`.

## Implementation

The three operators share one pattern, so this is a single phase. Port
`src/cuda/max_pool.cu`, `src/cuda/average_pool.cu`, and
`src/cuda/global_average_pool.cu` to OpenCL C.

Keep the host-side attribute, padding, and shape validation identical to the
verified CUDA rows. Compute in per-output-element kernels with a packed
`PoolParams` argument passed by value, following the OpenCL `Add` row.

`GlobalAveragePool` uses one work item per `(n, c)` row and accumulates in
double precision. Include the device fp64 capability check and the named error
path described in the requirements.

## Tests

For every operator, add a hand-computed case, a CPU-oracle comparison, and a
negative case. Cover the variants that matter:

- `MaxPool`: stride, padding, `ceil_mode`, dilation, and a missing
  `kernel_shape`;
- `AveragePool`: `count_include_pad` set to 0 and 1, a window that
  `ceil_mode` extends past the input, and the pad divisor rule;
- `GlobalAveragePool`: rank-4 and rank-3 inputs, plus a bad output shape and
  the fp64 capability guard.

## Acceptance

Run the default, CUDA, and OpenCL builds through their full CTest suites.
Check formatting and audit the diff against the expected specification and
source paths.

## Validation

Evidence is recorded in [`validation.md`](validation.md). Update the roadmap
and changelog when the work lands.
