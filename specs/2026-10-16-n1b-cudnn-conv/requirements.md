# Requirements — N1b cuDNN Convolution

Date: 2026-10-16. Parent packet: `specs/2026-10-16-n1-cuda/` (requirements
[R-1]–[R-4], hybrid vendor decision). Base: `main` post-N1a-M2.

## Goal

`Conv` on CUDA via cuDNN (v9 frontend/graph API per the backend
decision): same NCHW float32 semantics as the CPU oracle, plan cached at
startup, parity within tolerance on a GPU runner.

## In Scope

- [R-1] `Conv` CUDA kernel (`src/cuda/conv.cu`): groups, strides,
  dilations, explicit pads + `auto_pad` (NOTSET/VALID as in the loader
  policy), optional rank-1 bias. cuDNN v9 graph API with heuristics +
  per-shape autotune; engine plans cached per (dims, attrs) at first use.
- [R-2] `FindCudaKernel` gains `Conv`; loader unchanged
  (device-agnostic); executor staging already covers it.
- [R-3] Parity vs CPU oracle within `1e-3·(1+|ref|)` on: 1x1 and 3x3
  convs, stride-2, groups/depthwise, explicit pads, bias/no-bias —
  hand-computed small cases plus one multi-seed sweep.
- [R-4] Environment: develops in the repo `.devcontainer` (CUDA 12.8 +
  cuDNN devel); host has no cuDNN and stays cuDNN-free. The OFF build is
  unaffected.

## Out of Scope

- Other ops; FP16/BF16 paths (serving-stack concern; NERT stays
  float32-to-oracle); HIP/Vulkan ports (N1c/d); pre/post fusion (N1e).

## Constraints

- No `src/kernels`, loader, or shape changes; CPU oracle untouched.
- cuDNN is link-PRIVATE to `nert` when CUDA is on; OFF link closure
  stays stdlib-only.
