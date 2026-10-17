# Plan — N1g YOLO-on-GPU Closure

One worktree per batch (`wip/nert-n1g<n>`), host RTX 3060 verifies g1–g5;
the gate uses the `nert-gpu:26.08` container (cuDNN for Conv).

## g1 — reshape family (this slice)

- `src/cuda/reshape.cu`: full oracle validation mirror (allowzero, -1,
  shape-operand download via `device_to_host`), then D2D copy.
- `src/cuda/unsqueeze.cu`: axes-operand download, uniqueness/range checks,
  D2D copy. `src/cuda/flatten.cu`: axis attr, D2D copy.
- `src/cuda/expand.cu`: target-shape download + broadcastability checks,
  then a broadcast-copy kernel (Add-style stride template, f32/i64/bool).
- Table rows + `test/cuda_shape_test.cpp` (shared helpers header):
  Reshape with -1/allowzero, Unsqueeze negative axes, Flatten axis 0,
  Expand bcast incl. int64, negatives (bad product, dup axes, bad axis,
  non-broadcastable). Format; OFF + host-ON legs.

## g2–g5, gate — queued (own slices, same bars)

## Validation

Per-slice evidence lands in
`specs/2026-10-17-n1g-yolo-gpu/validation.md` — one table.
