# Plan — N1e First-Party CUDA Kernels

One worktree per batch (`wip/nert-n1e-<n>`), host RTX 3060 verifies every
slice. Container `nert-gpu:26.08` stays available; host suffices for
first-party work (no cuDNN needed).

## e1 — Mul/Sub/Div (this slice)

- `src/cuda/mul.cu`, `sub.cu`, `div.cu`: clone the `add.cu` broadcast
  template (helpers duplicated per file per house style); Div keeps IEEE
  zero-division (no throw, no guard).
- Table rows; `test/cuda_kernels_test.cpp` gains 3 exact tests (broadcast
  incl. int64) + CPU-vs-CUDA micro cases; format; OFF + host-ON legs.

## e2 / e3 — queued (own slices, same bars)

## Validation

Per-slice evidence lands in `specs/2026-10-16-n1-cuda/validation.md`
(N1e rows) — one table, no per-batch packet files.
