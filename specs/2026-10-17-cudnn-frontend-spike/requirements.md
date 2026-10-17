# Requirements — cuDNN Frontend spike (conv graph vs legacy row)

Date: 2026-10-17. The cuDNN Frontend (header-only C++ over the cuDNN
Graph API, Apache-2.0/MIT) is the candidate vehicle for N1e fusion and
a future N1b modernization. This packet authorizes a spike only — no
runtime change lands from it.

## Goal

Answer one question with measurements: can a frontend-built
convolution graph replace our legacy `cudnnConvolutionForward` row
with identical numerics and competitive timing?

## In scope

- [R-1] Build one convolution graph (ResNet-style 3x3 + a YOLO shape)
  with a pinned frontend release tag (never `develop`) against the
  container's cuDNN 9.25, from scratch sources outside the repo.
- [R-2] Parity vs the committed N1b row within `1e-3·(1+|ref|)`, timed
  on the RTX 3060.
- [R-3] A written verdict: adopt (with a follow-up packet) or drop,
  with reasons. Either outcome closes the spike.

## Out of scope

- Any change to `src/`, `include/`, or tests; vendoring the headers;
  fusion work itself (that gets its own packet on a positive verdict).

## Constraints

- Ampere reality: Hopper/Blackwell-only engines (Flash-Attention and
  friends) are out of scope on the RTX 3060; conv/matmul graphs only.
- NVIDIA-only by construction; HIP/OpenCL tracks are untouched.
- Host stays clean: everything happens in `nert-gpu:26.08`.
