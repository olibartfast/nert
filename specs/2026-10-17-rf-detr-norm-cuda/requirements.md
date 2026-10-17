# Requirements — RF-DETR norm corridor on CUDA (R1–R3 + ReduceSum)

Date: 2026-10-17. The RF-DETR seg-small inventory (task-models V-2) filed
CUDA gaps R1..R6b; this packet closes the four highest-leverage rows in
node-count order: R1 `LayerNormalization` (56 nodes), R2 `Sqrt` (48),
R3 `Erf` (17), R4 `ReduceSum` (5) — 126 nodes. Same model + same plan
on CUDA, matching the CPU oracle within tolerance.

## In scope

- Four first-party CUDA kernels: `src/cuda/sqrt.cu`, `src/cuda/erf.cu`,
  `src/cuda/reduce_sum.cu`, `src/cuda/layer_normalization.cu`, each
  mirroring its CPU oracle's validation policy exactly (dtype rules,
  attribute defaults, shape checks, error messages naming node + op).
- Table rows in `src/cuda/table.cu`, `target_sources` entries in
  `CMakeLists.txt`, tests in `test/cuda_activations_test.cpp` (Sqrt, Erf),
  `test/cuda_misc_test.cpp` (ReduceSum, beside ReduceMax), and a new
  `test/cuda_norm_test.cpp` + registration (LayerNormalization).
- Tests per op: hand-computed values, CPU-oracle vs CUDA compare at the
  bench tolerance, and negative load/shape cases. GPU-gated with
  `GTEST_SKIP` (never fail without a device), following the existing files.

## Out of scope

- HIP / OpenCL mirrors — queued. HIP follows via hipify-perl once this
  lands (Q-1); OpenCL is hand-written later in YOLO order.
- R5 `GridSample`, R6a (`Sin`/`Cos`/`Exp`), R6b (`Tile`/`Not`/`Greater`/
  `Less`/`And`) — separate slices. `Einsum` stays CPU-only (V-2).
- `bench/` fixtures (queued; `bench/` is under unrelated dirty work).
- Any change to `src/kernels/`, loader, shape inference, executor, or the
  Neuriplo-side parity harness.

## Constraints (traced to constitution)

- Default link closure stays standard-library-only; CUDA links PRIVATE.
- One device per graph chosen at load; no per-op fallback.
- Kernels naive and readable (one thread per element / per row, internal
  `cudaDeviceSynchronize` like the existing rows). Statistics and float
  sums accumulate in double, rounded once — like the CPU oracle and the
  Softmax precedent.
- Reduction rank limit 16 (ReduceMax precedent) unless the oracle allows
  less.

## Open questions

- Q-1 (task-models): where does each new GPU kernel live exactly once?
  **Decided 2026-10-17 (maintainer-directed): CUDA canonical;
  HIP via hipify-perl + hand-review; OpenCL hand-written.** This packet
  writes CUDA rows only; HIP mirrors are queued behind it.
- ReduceSum dtype scope: CPU supports float32 + int64. This packet ships
  both (templated accumulation: double for float, int64 for int64).
- LayerNormalization rank/axis policy mirrors the CPU kernel exactly
  (trailing axes from `axis`, default -1; Scale/B per-element or scalar;
  1–3 outputs with optional Mean/InvStdDev); no new policy invented here.
