# CUDA Operator Closure

Date: 2026-10-18.

## Purpose

The CUDA table covered nearly every CPU operator. Three were still missing:
`GridSample` (four RF-DETR nodes, previously deferred as work item `R5`),
`ConvTranspose` (YOLO segmentation), and `InstanceNormalization` (RAFT). This
specification adds all three. Every CPU operator is now available on CUDA,
including `Einsum`, whose CUDA row was added during the OWLv2 bring-up. The
earlier `V-2` check treated `Einsum` as CPU-only; that exception is historical.

## GridSample

The first-party kernel samples a four-dimensional input at normalized grid
positions. It supports bilinear and nearest sampling, `zeros`, `border`, and
`reflection` padding, and `align_corners` 0 or 1. Coordinate arithmetic matches
the CPU oracle in double precision; one thread handles each `(n, ho, wo)`
position and loops over channels. Nearest sampling uses round-half-even.

Implementation: `src/cuda/grid_sample.cu`; tests:
`test/cuda_sample_test.cpp`.

## InstanceNormalization

The first-party kernel normalizes each `(n, c)` row over its spatial elements.
Statistics accumulate in double precision, scale and bias are rank-1 `[C]`,
and epsilon defaults to `1e-5`. It follows `LayerNormalization`'s
one-thread-per-row structure with per-channel parameters.

Implementation: `src/cuda/instance_normalization.cu`; tests:
`test/cuda_norm_test.cpp`.

## ConvTranspose

`ConvTranspose` uses cuDNN and is available only with `NERT_HAS_CUDNN`. It
follows the existing cuDNN `Conv` structure: `cudnnConvolutionBackwardData`, v7
heuristics, a per-geometry plan cache, a first-party bias add, and
`CUDNN_FMA_MATH` so TensorFloat-32 (TF32) is forbidden.

It accepts rank-4 NCHW tensors and explicit padding or `VALID`; other automatic
padding modes are rejected by the loader, matching the CPU oracle. Non-zero
`output_padding` and asymmetric padding are rejected on CUDA because cuDNN
does not provide those knobs. The CPU oracle accepts both, so these are
deliberate CUDA-narrower policies. The implementation, table row, and
`target_sources` entry are guarded by `NERT_HAS_CUDNN`; tests live in
`test/cuda_conv_test.cpp` behind `HasCudaConvTranspose()`.

## Boundaries

Out of scope: HIP and OpenCL mirrors, bench fixtures, loader and
shape-inference changes, executor changes, Neuriplo parity runs, and per-call
algorithm or workspace selection.

The default build stays standard-library-only; CUDA and cuDNN link privately.
A graph uses one device. First-party kernels stay naive and readable, with
internal synchronization and double-precision accumulation. Packed-index rows
retain the rank cap of 16. The new rejections name the node and operator and
are the only deliberate CUDA-narrower-than-CPU policies here.

## Open Questions

None. The maintainer approved the scope and the cuDNN-gated `ConvTranspose`
approach on 2026-10-18. `Q-1`, the GPU single-source question, is unchanged.

This host has no system cuDNN. The `NERT_HAS_CUDNN` build is verified through
`CUDNN_ROOT` pointing at a local symlink farm for the pip cuDNN 9 package; the
farm is a verification aid, not a repository change.
