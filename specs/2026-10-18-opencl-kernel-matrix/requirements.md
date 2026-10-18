# Requirements — missing-model kernel matrix (maintainer directive)

Date: 2026-10-18. Take every Neuriplo task model still missing from the
NERT coverage matrix to a verified device gate, implementing only the
rows its export actually uses.

## Pre-closed (do not re-open)

- **P0 zero-extent Concat** on CPU/CUDA/OpenCL: closed in `329c2e7`
  with `[2,0,4] + [2,3,4]` regression tests on all three families.
- **bool Concat** (OpenCL): same commit.
- **GlobalAveragePool fp64**: explicit `CL_DEVICE_DOUBLE_FP_CONFIG`
  gate with a named error (the directive's fp64-only option), same
  commit.
- **Conv / Flatten / Gemm** on OpenCL: `3be01a6`, `9bfab64`. ResNet-18
  is op-complete on OpenCL (OPS, gate pending).
- **opset-18 conversions**: `7434da0`. 13 new loadable exports; CPU and
  CUDA op-complete. Grounding DINO loads at 19,224 folded nodes.
- **RF-DETR CUDA full-model gate**: PASS (`f678940`), boxes 9.938e-04,
  labels 3.080e-04, top-1 100%, mask IoU 0.999528.

## In scope

- OpenCL rows still missing for the loadable models, in the directive's
  order: MatMul, LayerNormalization, Cast, Where, Expand,
  ConstantOfShape, Equal, Erf, Split, GatherElements, Resize, TopK, And,
  Greater, LessOrEqual, Einsum, ArgMax, Elu, ReduceL2, Sqrt, ReduceSum,
  Cos, Sin, Exp, Not, Tile, ReduceMax, Less, Mod.
- Each row: hand-computed case, CPU-oracle comparison, negative case,
  dtype variants per the oracle (no silent float-only narrowing), and a
  full OpenCL leg run.
- Full-model gates in order: ResNet-18, YOLO26n, OWLv2, RF-DETR
  seg-small on OpenCL; then converted models by node count; RAFT and
  Grounding DINO after the transformer rows.
- Rebuild measured coverage after each lane merge.

## Out of scope / recorded blockers

- **CUDA rows**: none unless a newly converted export proves a real gap.
  All current CPU op types have CUDA rows.
- **HIP**: no `hipify-perl`, no ROCm toolchain and no AMD hardware on
  this host. HIP stays at 9 rows; every HIP model cell is BLOCKED until
  a HIP host exists. No fabricated HIP results.
- **DETR (RT-DETR/D-FINE/DEIM/DEIMv2), ViT/ViTPose,
  Depth-Anything-V2, VideoMAE, ViViT, TimeSformer**: no local export
  (searched 2026-10-18). No rows invented; provisioning is a prerequisite.
- **RAFT**: real torchvision RAFT-large export is being provisioned
  (Lane A); the local `ref.onnx` is a FoundationPose refiner and is not
  RAFT.

## PASS rule

A model cell is PASS only from a full-model run against the reference on
that device. Kernel-row coverage is OPS. A row landing is not a gate.
