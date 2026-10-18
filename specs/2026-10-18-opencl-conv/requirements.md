# Requirements — OpenCL Conv

Date: 2026-10-18. The last shared portable row and the hard dependency
for every loadable model. Port the verified CPU/CUDA semantics to a
first-party OpenCL C kernel.

## In scope

- `src/opencl/conv.cpp`: ONNX cross-correlation, input `[N,C,spatial...]`,
  weight `[M,C/group,k...]`, optional bias `[M]`, groups, strides,
  dilations, explicit `pads`, `auto_pad` NOTSET/VALID/SAME_UPPER/
  SAME_LOWER, rank 3/4/5 (1-D/2-D/3-D) lifted to three spatial axes.
  Float accumulation with one rounding; one work-item per output.
- Table row, `CMakeLists.txt` source, tests in
  `test/opencl_kernels_test.cpp` (hand-computed 2-D/grouped/1-D,
  stride+pad+bias oracle compare, negatives).
- The OpenCL no-row probe moves to `Sin` (Conv now has a row).

## Out of scope

- HIP mirror (no hipify on this host), performance tuning/tiling,
  `bench/`, executor/loader changes, Neuriplo runs.

## Constraints

- stdlib-only default; OpenCL PRIVATE, opt-in. Mirror the CPU oracle's
  validation and messages exactly. `clFinish` sync.

## Open questions

- None.
