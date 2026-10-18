# Requirements — OpenCL Flatten + Gemm

Date: 2026-10-18. Follow-up to the OpenCL movement six and pools. These
two rows take ResNet-18's portable gap down to `Conv` alone and close the
`Flatten`/`Gemm` lines for YOLO/OWLv2/RF-DETR.

## In scope

- `src/opencl/flatten.cpp`: validated flat buffer copy (f32/i64/bool),
  `axis` default 1 with negative wrap from `rank + 1`, mirroring
  `src/kernels/flatten.cpp` validation and messages.
- `src/opencl/gemm.cpp`: `Y = alpha * op(A) * op(B) + beta * C`,
  row-major, `transA`/`transB`, bias `[N]` or `[M,N]`, float
  accumulation, one work-item per output. Mirror
  `src/kernels/gemm.cpp` validation/messages and shape rules.
- Table rows, `CMakeLists.txt` sources, tests in
  `test/opencl_kernels_test.cpp` (hand-computed + CPU-oracle compare +
  negatives, `GTEST_SKIP` without a platform).

## Out of scope

- `Conv` (next slice), HIP mirrors, `bench/`, loader/executor changes,
  Neuriplo runs, docs handled by the orchestrator (coverage/CHANGELOG).

## Constraints

- stdlib-only default; OpenCL PRIVATE, opt-in. `clFinish` sync as in the
  existing rows. No fp64 requirement for Gemm (float accumulation,
  tolerance-compared); the oracle accumulates in double.

## Open questions

- None.
