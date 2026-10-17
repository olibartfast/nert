# Plan — Other task models on NERT

One worktree per phase (`wip/nert-task-<n>`). Planner commits touch
main only; code merges only on full validation evidence. Q-1 (kernel
single-source home) must be answered before any phase writes a new
GPU row — CPU parity work can start regardless.

## Phase 1 — ResNet-18 on CUDA (classification)

Dispatch the CI fixture model on the CUDA device and run parity
against ORT. Expectation: green with no new kernels, since Conv, Gemm,
Relu, and both pools already have CUDA rows. Any surprise gap becomes
its own mini-slice, YOLO-ordered.

## Phase 2 — DETR gap analysis (transformer detection)

Once the export and harness exist (Q-2): inventory every node and op,
record which lack GPU rows (LayerNorm is already known), and file the
missing rows as N1 slices in node-count order. No implementation here —
this phase delivers the inventory and the slice list.

## Phase 3 — Open-vocab parity (OWLv2, Grounding-DINO)

CPU-vs-ORT parity first, exercising the Tier-5 kernels; then the same
dispatch-gap treatment as Phase 2 for the GPU side.

## Phase 4 — Living coverage matrix

A short document (next to this packet) crossing models with devices,
updated by every slice that lands. A missing check is a roadmap row
with an owner — never a silent hole.

## Validation

Per-phase evidence lands in
`specs/2026-10-17-task-models/validation.md` as it is measured.
