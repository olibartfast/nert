# Plan — N1e via cuDNN Frontend

One worktree per slice (`wip/nert-n1e-a`, `...-b`, `...-c`); merge each
on green evidence. The spike program stays scratch; only shipped code
and tests land.

## Slice A — cached legacy Conv state

1. Extend the plan cache entry with the tensor/filter/convolution
   descriptors and a process-lifetime handle. Keep the geometry key.
2. Move descriptor creation into the cache fill; reuse on every call.
3. Bench `Conv` before/after (same fixture, three runs) + parity exact.

## Slice B — frontend Conv row

1. CMake: `NERT_CUDNN_FRONTEND_DIR` cache path; when set and CUDA is
   on, add `src/cuda/conv_frontend.cu`, include the frontend root, link
   `nvrtc`.
2. Build one graph per geometry (validate → operation graph → plans →
   support → build plans → workspace), cached like A. Execute with the
   executor's buffers; pin FLOAT io/compute and contiguous NCHW output.
3. Tests: graph vs CPU oracle (ResNet + YOLO shapes, stride/pad
   variants), graph vs legacy row bit-exact, missing frontend dir keeps
   the legacy row (default OFF and host CUDA legs).
4. Bench `Conv` with the frontend row; record both.

## Slice C — fused Conv bias

1. Add the bias tensor to the cached graph when present (dims
   `{1, C, 1, ...}`, stride `{C, 1, C, ...}` like the frontend samples)
   and a pointwise ADD node; no separate bias kernel.
2. Key the cache on bias presence; pack the bias pointer per call.
3. Compare against the sliced row on the padded/grouped/bias fixtures;
   bench the bias fixture before/after.

Activation fusion (Conv→Relu as separate nodes) needs executor pattern
matching and is out of scope here.

## Validation

Evidence lands in `specs/2026-10-18-n1e-cudnn-graph/validation.md`,
one row per slice, with the bench table and the exact commands.
