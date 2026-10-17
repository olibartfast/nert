# Validation — N1g YOLO-on-GPU Closure

Defined before implementation. Bars: OFF untouched, format clean, no
`src/kernels`/loader/shape/executor diffs.

## Checks

- [V-1] Per batch: hand-computed vs-oracle tests + negatives, ON matrix
  green on the host leg, OFF 8/8, format clean.
- [V-2] Gate: yolo26n loads + runs on the CUDA leg (cuDNN container),
  confident-row agreement vs CPU at the V-5a bar (min IoU ≥ 0.999998,
  score diff ≤ 1e-4, classes match).
- [V-3] No-row probe discipline: the `MatMul` probe in
  `cuda_kernels_test.cpp` moves to the next uncovered op when g4 lands.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| N1g-1 | Reshape/Unsqueeze/Flatten/Expand CUDA vs CPU (-1/allowzero/neg-axes/bcast/i64, 4 negatives) + OFF 8/8 + ON 14/14 + format | pass | 2026-10-17 | RTX 3060 host leg; exact match. YOLO coverage now 301/485 (62.1%). |
