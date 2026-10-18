# Plan — CUDA Operator Closure

Work in the main tree. Do not modify `bench/`,
`specs/2026-10-17-cudnn-frontend-spike/`, or `.github/`.

## GridSample

Port the CPU oracle's `SourceCoordinate`, `Reflect`, and `at` helpers to
`__device__` functions in double precision. Use grid-stride threads over
`(n, ho, wo)` positions, with an inner loop over channels.

Tests cover bilinear interior and zero-padded edges, border and reflection
padding, nearest half-even, `align_corners=1`, and negatives for an invalid
mode, an invalid padding mode, a batch mismatch, and a non-4D input.

## InstanceNormalization

Follow `layer_normalization.cu`: one thread per `(n, c)` row, double-precision
statistics, and rank-1 `[C]` scale and bias validated with the `ChannelParam`
rule. Tests in `test/cuda_norm_test.cpp` cover a basic case, a rank-2 input,
and negatives for a bad scale size and a rank-1 input.

## ConvTranspose

Copy the structure of `src/cuda/conv.cu`: attribute parsing, NCHW validation,
the oracle output-shape check, a rank-1 `[M]` bias and first-party bias kernel,
a per-geometry plan cache, `cudnnGetConvolutionBackwardDataAlgorithm_v7`,
workspace allocated once per key, and `CUDNN_FMA_MATH` so TensorFloat-32
(TF32) is forbidden.

Mapping for the backward-data call:

```text
dyDesc = [N, C, H, W]        (input X)
wDesc  = [C, M/group, kH, kW]
dxDesc = [N, M, Ho, Wo]      (output Y)
```

Reject non-zero `output_padding` and asymmetric padding with clear errors.
Tests in `test/cuda_conv_test.cpp`, behind `HasCudaConvTranspose()`, cover a
2x2 stride-2 no-bias case with hand-computed output, a 3x3 stride-1 pad-1 case
with bias compared with the oracle, and negatives for a bad output shape and
rejected `output_padding`.

## Acceptance

Each operator needs a hand-computed case, an oracle comparison, and negatives.
Run the targeted target during implementation, then the full matrix: default
build, CUDA build, cuDNN build via the local `CUDNN_ROOT` farm, format check,
and a diff audit limited to the expected paths.

## Validation

Evidence is recorded in [`validation.md`](validation.md). Update the roadmap
and changelog when the change lands.
