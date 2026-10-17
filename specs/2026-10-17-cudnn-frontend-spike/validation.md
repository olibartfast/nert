# Validation — cuDNN Frontend spike

Defined before implementation. Bar: parity within `1e-3·(1+|ref|)` vs
the N1b row, timings on the RTX 3060, verdict recorded either way.

## Checks

- [V-1] Frontend graph builds + runs against cuDNN 9.25 in the
  container from a pinned release tag.
- [V-2] Parity + timing table vs the legacy row for both shapes.
- [V-3] Adopt-or-drop verdict with reasons.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | `nvcc` spike in `nert-gpu:26.08` against cuDNN 9.25.0, frontend `v1.31.0` (release tag, shallow clone; header-only; links `-lcudnn -lcudart -lnvrtc`) | pass | 2026-10-18 | Runs on Ampere RTX 3060; graph build + plans + execute all good. |
| V-2 | Conv graph vs committed N1b legacy row, same device buffers, 10 synchronized iterations | pass | 2026-10-18 | See table below; bit-identical (maxdiff 0.0). |
| V-3 | Verdict | adopt | 2026-10-18 | N1e fusion vehicle; follow-up packet for a cached-graph Conv row. |

Parity + timing (RTX 3060, cudnn 92500, frontend 13100):

| shape | frontend ms | legacy ms | ratio | maxdiff | over tolerance |
|-------|-------------|-----------|-------|---------|----------------|
| ResNet-style x[1,64,56,56] w[64,64,3,3] s1 p1 | 0.053 | 2.090 | 39.3x | 0.000e+00 | 0/200704 |
| YOLO x[1,3,640,640] w[32,3,3,3] s2 p1 | 0.112 | 2.259 | 20.3x | 0.000e+00 | 0/3276800 |

Both runs force FLOAT io/compute and a contiguous-NCHW output stride, so
the raw buffers are directly comparable. The ratio is end-to-end as the
runtime calls Conv today: the legacy row creates a cuDNN handle +
descriptors, runs heuristics, and synchronizes on every call, while the
frontend reuses a built graph + plans + workspace (one-time build cost
excluded). The honest conclusion is both a parity and a per-call
overhead win; kernel-level tuning is a separate question.

## Verdict: adopt

Reasons:

1. **Bit-identical numerics** to the committed row on both shapes, so it
   can replace or back the CUDA Conv path without a tolerance change.
2. **Fusion** is the actual N1e need: Conv+Bias+Activation and
   MatMul+epilogue graphs are native, without hand-rolled kernels.
3. **Pinned release works**: `v1.31.0` builds header-only against the
   container's cuDNN 9.25; no vendoring, no `develop` branch.

Caveats for the follow-up packet:

- Ampere reality: conv/matmul graphs only; Hopper/Blackwell attention/MoE
  engines are out of scope on the RTX 3060.
- NVIDIA-only by construction; HIP/OpenCL tracks are untouched (Q-1).
- Frontend needs `nvrtc` at link time and ships nlohmann/json in its
  headers; both stay in the CUDA-only link closure.
- Adopt behind `NERT_ENABLE_CUDA`; pin `v1.31.0` (never `develop`).
- Decide whether N1e replaces the legacy `cudnnConvolutionForward` row
  with a cached frontend graph or keeps both during a transition.
