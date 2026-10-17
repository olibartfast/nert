# Requirements — N1g YOLO-on-GPU Closure (ordered batches)
Date: 2026-10-17. Base: `main` post-N1f. Gate model: YOLO26n opset 18
(`neuriplo/build-parity/native_yolo_parity_fixture/yolo26n.onnx`, 485
nodes) — CPU parity already evidenced (V-5a, min IoU 0.9999987).

## Problem

YOLO26n on the CUDA table: 276/485 nodes covered (56.9%); first blocker
is `/model.0/conv/Conv` (no cuDNN on the host leg), then 107 more
missing nodes. Missing histogram: Conv 102, Concat 28, Reshape 15,
Split 12, Unsqueeze 11, Expand 8, Transpose 6, MatMul 4, GatherElements 3,
Cast 3, Resize/Slice/TopK/ConstantOfShape/Equal/Where 2 each,
Shape/Gather/ReduceMax/Flatten/Mod/Sub 1 each.

## Goal

Every YOLO26n op runs on CUDA; the gate is yolo26n loading + running on
the GPU leg with confident-row agreement vs the CPU oracle (same bar as
V-5a). Batches stay YOLO-ordered (highest node count first).

## Batches

- [R-1] **g1 (reshape family, 25 nodes):** `Reshape`/`Unsqueeze`/`Flatten`
  (validate on host + `cudaMemcpy` device-to-device; dtypes per oracle:
  f32/i64/bool for Reshape, f32/i64 for the other two) + `Expand`
  (broadcast-copy kernel, f32/i64/bool). Shape/axes operands live on
  device and are downloaded for validation (tiny, one copy).
- [R-2] **g2 (concat family, 42 nodes):** `Concat`/`Split`/`Slice` —
  per-output-element index kernels, negatives naming node + op.
- [R-3] **g3 (transpose/gather, 10 nodes):** `Transpose` (+ `Gather`/
  `GatherElements`) — permuted/index copy kernels.
- [R-4] **g4 (matmul, 4 nodes):** `MatMul` via cuBLAS (same row-major swap
  trick as `Gemm`).
- [R-5] **g5 (misc, 16 nodes):** `Cast`/`TopK`/`Where`/`Equal`/`Shape`/
  `ConstantOfShape`/`ReduceMax`/`Mod`/`Resize` (+ `Squeeze` only if cheap —
  0 YOLO nodes, so no).
- [R-6] **Gate:** yolo26n runs on the CUDA leg (cuDNN container for Conv;
  first-party everywhere else) with V-5a-grade agreement. Conv itself
  stays cuDNN per N1b — no first-party Conv.

## Out of Scope

- Non-YOLO ops; HIP/Vulkan; fusion; deleting legacy bench (N1d follow-up).

## Constraints

- Same bars: OFF untouched, format clean, no `src/kernels`/loader/shape/
  executor diffs, tolerance `1e-3·(1+|ref|)`, SKIP contract intact.
- g1–g5 verify on the host RTX 3060 leg (no cuDNN needed); the gate runs
  in `nert-gpu:26.08` (cuDNN present).
