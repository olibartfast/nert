# Requirements — CUDA closure: GridSample + ConvTranspose + InstanceNormalization

Date: 2026-10-18. The CUDA table is complete except `GridSample` (R5,
4 RF-DETR nodes, explicitly deferred "own slice"),
`ConvTranspose` (Tier-1 YOLO/segmentation), and
`InstanceNormalization` (Tier-4 RAFT). This packet lands all three.
After it, every CPU op is on CUDA except `Einsum` (stays CPU-only per
V-2) — full closure.

## In scope

- `src/cuda/grid_sample.cu` (first-party): bilinear + nearest,
  zeros/border/reflection, align_corners 0/1 — the oracle's exact
  coordinate math in double, one thread per (n, ho, wo) position with
  an inner channel loop. Nearest uses round-half-even (`nearbyint`).
- `src/cuda/instance_normalization.cu` (first-party): per-(n,c)
  double-accum rows, rank-1 [C] scale/B, epsilon default 1e-5 —
  the LayerNormalization pattern with per-channel params.
- `src/cuda/conv_transpose.cu` (cuDNN, `NERT_HAS_CUDNN` only):
  `cudnnConvolutionBackwardData` legacy path (descriptors + v7
  heuristics + first-party bias add), `CUDNN_DEFAULT_MATH`
  (TF32 forbidden), per-geometry plan cache — the N1b Conv structure.
  Rank-4 only, explicit pads or VALID (loader rejects the rest, like
  the oracle). Non-zero `output_padding` and asymmetric pads are
  rejected on CUDA with a clear message (cuDNN has neither knob);
  the oracle still accepts them on CPU.
- Table rows (ConvTranspose inside `#ifdef NERT_HAS_CUDNN`),
  `target_sources` (same guard), tests: new `test/cuda_sample_test.cpp`
  + target (GridSample), `test/cuda_norm_test.cpp` (InstanceNorm),
  `test/cuda_conv_test.cpp` (ConvTranspose, `HasCudaConvTranspose`
  guard mirroring `HasCudaConv`).

## Out of scope

- HIP/OpenCL mirrors (queued hipify/hand-write split), `bench/`
  fixtures, loader/shape/executor changes, Neuriplo parity runs,
  per-call algo search / per-call workspace (documented N1b follow-ups).

## Constraints (traced to constitution)

- stdlib-only default; CUDA/cuDNN PRIVATE. One device per graph.
- Naive readable first-party kernels, internal sync, double coordinate
  and stat accumulation. Rank cap 16 on the packed-index first-party
  rows (established pilot limit).
- New rejections (output_padding, asymmetric pads) name node + op and
  are recorded here as the only deliberate CUDA-narrower-than-CPU
  policies in this packet.

## Open questions

- None open: scope (all three) and ConvTranspose approach (cuDNN-gated)
  decided 2026-10-18 (maintainer-directed). Q-1 unchanged.
- Local verification note: this host has no system cuDNN; the
  NERT_HAS_CUDNN leg is verified via `CUDNN_ROOT` pointed at a
  symlink farm to the pip cuDNN 9 package (local-only, no repo change).
