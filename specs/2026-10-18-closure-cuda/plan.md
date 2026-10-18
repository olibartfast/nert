# Plan — CUDA closure: GridSample + ConvTranspose + InstanceNormalization

Work in the main tree. Forbidden: `bench/`, `specs/2026-10-17-cudnn-frontend-spike/`,
`.github/`. `implementer` dispatch still unavailable (harness);
planner implements directly.

## Phase A — GridSample (first-party)

Pattern: Resize-linear precedent (device coordinate math) + the oracle's
`SourceCoordinate`/`Reflect`/`at` in double, ported verbatim as
`__device__` helpers. Grid-stride threads over (n, ho, wo); inner loop
over channels. Tests: bilinear interior + zeros edge, border, reflection,
nearest half-even, align_corners=1, negatives (bad mode, bad padding,
batch mismatch, rank).

## Phase B — InstanceNormalization (first-party)

Pattern: `layer_normalization.cu` one-thread-per-row, double accum;
row = (n, c), inner = spatial product; scale/bias rank-1 [C] validated
(`ChannelParam` rule). Tests in `test/cuda_norm_test.cpp`: basic + rank-2
(single spatial) + negatives (bad scale size, rank-1 input).

## Phase C — ConvTranspose (cuDNN bwd-data, gated)

Structure copied from `src/cuda/conv.cu`: attribute parsing with
ConvTranspose messages, NCHW validation, oracle output-shape formula
check, bias rank-1 [M] + first-party bias kernel, plan cache keyed by
geometry, v7 `cudnnGetConvolutionBackwardDataAlgorithm_v7`, workspace
once per key, `CUDNN_DEFAULT_MATH`. Mapping: dyDesc=[N,C,H,W] (our X),
wDesc=[C,M/group,kH,kW], dxDesc=[N,M,Ho,Wo] (our Y). Extra rejections:
non-zero output_padding, asymmetric pads. Tests in
`test/cuda_conv_test.cpp` behind `HasCudaConvTranspose()`:
2x2 stride-2 no-bias hand-computed, 3x3 stride-1 pad-1 with bias vs
oracle, negatives (bad output shape, output_padding reject).

## Per-phase bars

Hand-computed + oracle-compare + negatives; targeted target green;
final acceptance once at end: OFF all-green + CUDA-ON all-green
(+ cuDNN leg via local CUDNN_ROOT farm) + format clean + diff audit
= packet paths only.

## Validation

Evidence in `specs/2026-10-18-closure-cuda/validation.md`.
Roadmap + CHANGELOG sync on landing (planner-owned).
