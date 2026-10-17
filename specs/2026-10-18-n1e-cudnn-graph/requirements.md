# Requirements — N1e via cuDNN Frontend (cached Conv, then fusion)

Date: 2026-10-18. Follow-up to the cuDNN Frontend spike
(`specs/2026-10-17-cudnn-frontend-spike/`, verdict: adopt v1.31.0).
Parent: `specs/2026-10-16-n1e-first-party/` (fusion was out of scope
there). This packet turns the spike into shipped behavior in slices,
each with its own evidence.

## Problem

The spike showed two things on the RTX 3060:

1. A frontend Conv graph is **bit-identical** to the committed N1b row
   on ResNet- and YOLO-shaped convolutions.
2. The committed row costs ~2.1-2.3 ms per call, while the built graph
   executes in 0.05-0.11 ms. Most of that gap is per-call setup: the
   row creates a cuDNN handle and descriptors, runs heuristics, and
   synchronizes on every call.

Fusion (Conv+Bias+Activation, MatMul+epilogue) remains unimplemented,
which is the actual N1e goal.

## Slices

- [R-1] **A — cache legacy Conv state, no new dependency.** Keep one
  process-lifetime cuDNN handle and cache descriptors + algo + workspace
  per resolved geometry (extend the existing plan cache). Behaviour and
  numerics unchanged. Evidence: Conv bench row before/after, parity
  exact, full CUDA + OFF legs green.
- [R-2] **B — frontend Conv row.** Add an opt-in
  `NERT_CUDNN_FRONTEND_DIR` (header-only include root) behind
  `NERT_ENABLE_CUDA`; when present, compile `src/cuda/conv_frontend.cu`
  and use a cached `fe::graph::Graph` for Conv, keeping the legacy row
  as the fallback when the directory is absent. Evidence: bit-exact vs
  the CPU oracle and vs the legacy row, bench row, container + host
  legs, default-OFF untouched.
- [R-3] **C — fused Conv+Bias+Activation.** One frontend graph for the
  ResNet/YOLO corridor (Conv → bias add → Relu), compared against the
  composed legacy rows; bench proves the fused win and no tolerance
  change.
- [R-4] **Pinned dependency.** `v1.31.0` only (never `develop`), no
  vendored headers, nothing in the OFF link closure.

## Out of scope

- HIP/OpenCL mirrors (Q-1: CUDA canonical, HIP via hipify, OpenCL
  hand-written; their slices follow separately).
- Attention/MoE frontend engines (Hopper/Blackwell-only; Ampere box).
- Changing the planner, loader, shape inference, or executor.
- MatMul epilogue fusion (own slice after C, same mechanism).

## Constraints

- Default build stays stdlib-only; frontend headers and `nvrtc` link
  only inside `NERT_ENABLE_CUDA` when R-2 is enabled.
- One device per graph; no runtime fallback between backends.
- Tolerance unchanged: `1e-3·(1+|ref|)`; the spike achieved 0.0.
- No `src/kernels/`/loader/shape diffs.

## Open question

- Dependency sourcing for B: configure-time `NERT_CUDNN_FRONTEND_DIR`
  (offline-friendly, default empty) vs `FetchContent` at a pinned tag
  (one-command builds, needs network). Recommendation: support the
  cache variable first, add FetchContent as an opt-in convenience.
