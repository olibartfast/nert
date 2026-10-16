# Requirements — N1e First-Party CUDA Kernels (incremental batches)

Date: 2026-10-16. Parent packet: `specs/2026-10-16-n1-cuda/` (hybrid
decision: first-party where libraries don't help). Base: `main` post-N1b.

## Goal

Grow CUDA coverage op by op with first-party kernels for everything
outside cuBLAS/cuDNN territory, each proven against the CPU oracle on
the RTX 3060. N1c (HIP) later hipifies these sources unchanged in
structure.

## Batches

- [R-1] **e1 (elementwise arithmetic):** `Mul`/`Sub`/`Div` — same
  broadcast template as CUDA `Add` (float32/int64, rank cap 16);
  division-by-zero follows IEEE (never throws), matching the oracle.
- [R-2] **e2 (pooling):** `MaxPool` (+ `AveragePool` if cheap) — NCHW,
  kernel_shape/strides/pads/dilations/ceil_mode, indices output
  omitted exactly like the oracle (primary output only).
- [R-3] **e3 (activations/normalization):** `Sigmoid`/`Tanh`/`Softmax`
  (numerically stable, double accumulation where the oracle does),
  `BatchNormalization` if the zoo needs it (check CPU coverage first —
  ResNet-18's BN folds at export, so e3 may skip it with a note).
- Each batch: kernel(s) + table rows + hand-computed vs-oracle tests +
  one Model-level graph where it adds staging proof. No loader/shape/
  executor changes (staging covers all).

## Out of Scope

- Library territory (Gemm/Conv/MatMul eventually via cuBLAS — separate
  slice if needed); HIP/Vulkan ports; fusion beyond what the oracle
  already fuses; pre/post pipeline kernels (own packet when scoped).

## Constraints

- Same bars: OFF untouched, format clean, no `src/kernels`/loader/
  shape diffs, tolerance `1e-3·(1+|ref|)`, SKIP contract intact.
