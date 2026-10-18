# Plan — OpenCL Flatten + Gemm

Single phase, two rows, one pattern each.

- Flatten: host-side axis/shape/dtype validation copied from the CPU
  row, then `clEnqueueCopyBuffer` of the full buffer (order never
  changes) + `clFinish`.
- Gemm: host-side dtype/rank/transpose/contraction/bias validation
  copied from the CPU row; one OpenCL C kernel, one work-item per
  `(m, n)` output, `float` accumulator with one rounding; by-value
  params struct (Add precedent); global size `m*n`.
- Tests: Flatten axis 0/1/-1 and f32/int64/bool; Gemm hand-computed
  2x2, transposes, alpha/beta, both bias ranks, plus negatives.
- Docs (coverage, CHANGELOG, table counts) are orchestrator-owned after
  review.

Bars: OFF + CUDA + OpenCL legs green; format clean; diff limited to the
packet paths.

## Validation

Evidence in `validation.md`.
