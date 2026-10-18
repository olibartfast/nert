# Validation — Task-Model Coverage on NERT

Defined before implementation. The checks below prove model behavior, not just
operator coverage.

## Acceptance Rules

- Classification must match ONNX Runtime's top-1 result; the score difference
  is recorded and must stay within the agreed tolerance.
- GPU deviations follow the YOLO26n precedent, recorded as gate `R-6`: the
  difference is measured, the detection result is identical, and the deviation
  is noted by the maintainer.
- A model counts as passing only after a full-model run against the reference.
  A complete operator table is useful evidence, but it is not a gate.

## Checks

| ID | What is checked | Passing evidence |
| --- | --- | --- |
| `V-1` | ResNet-18 on CUDA: top-1 matches ONNX Runtime and the score difference is recorded. | CUDA run with all 49 nodes dispatched; top-1 `107 = 107`, top-5 order identical, maximum absolute difference `2.167e-03`. |
| `V-2` | DETR: the export is inventoried and its missing GPU operators are listed. | RF-DETR seg-small inventory completed; CPU covers every non-folded operator; CUDA gaps became work items R1–R6b. |
| `V-3` | OWLv2 and Grounding DINO: CPU parity is recorded and GPU gaps are listed. | OWLv2 CPU and CUDA gates passed; Grounding DINO still needs a full reference gate. |
| `V-4` | The model-by-device coverage matrix exists and is current. | First measured pass recorded in [`coverage.md`](coverage.md); update it as rows and gates land. |

## Evidence

| ID | Check | Result | Date | Notes |
| --- | --- | --- | --- | --- |
| `V-1` | ResNet-18 CUDA gate in `nert-gpu:26.08` (cuDNN) against ONNX Runtime 1.27.0 | Pass with recorded deviation | 2026-10-17 | Top-1 `107 = 107`; top-5 order identical; maximum absolute difference `2.167e-03`. The run took 1094 ms and dispatched all 49 nodes, proving run-to-completion. Deviation: twenty cuDNN convolutions compound float accumulation differently from the ONNX Runtime CPU kernels, but the classification result is identical, matching the `R-6` precedent. Export: torchvision 0.27, opset 18, seed 0. |
| `T1a` | `GlobalAveragePool` CPU and CUDA rows: exact channel means, rank-1 negative, shape collapse, loader allowlist; OFF build 8/8; host CUDA GAP parity; format | Pass | 2026-10-17 | This was the only operator missing from the 49-node ResNet-18 inventory. Double accumulation is bit-exact on both sides. |
| `V-2` | RF-DETR seg-small inventory: 1784 nodes, opset 17 converted bit-exactly to 18; CPU covers all non-folded operators; CUDA gaps filed as R1–R6b | Pass | 2026-10-17 | Full CPU run completed in 571 s with all nodes dispatched. CPU versus ONNX Runtime: boxes `5.2e-04`, labels top-1 100%, binarized mask IoU minimum `1.000000` — model-identical. Work items: R1 `LayerNormalization` (56), R2 `Sqrt` (48), R3 `Erf` (17), R4 `ReduceSum` (5), R5 `GridSample` (4), R6a `Sin`/`Cos`/`Exp` (11), R6b `Tile`/`Not`/`Greater`/`Less`/`And` (9). `Einsum` was CPU-only at inventory time; the later OWLv2 bring-up added a CUDA row (`V-3b`). |
| `V-3` | OWLv2-base-patch16 CPU parity: 1219 nodes, native opset 18, NERT-side weights | Pass | 2026-10-18 | Logits are bit-identical (top-1 `3600/3600`); predicted boxes differ by at most `1.297e-04`; no row exceeds tolerance. The Release run took 3037 s. This also answers `Q-2` for the open-vocab export. CUDA gaps overlap the RF-DETR work items (LayerNorm R1, Erf R3) plus `IsNaN` (12), `ReduceL2` (3), and `And`/`LessOrEqual`/`ArgMax`/`Greater`/`Elu` (6). `Einsum` was CPU-only at that point; the later CUDA row is recorded in `V-3b`. |
| `V-3a` | OWLv2 CPU bring-up fixes found by the real model | Pass | 2026-10-18 | Fixed int32 `Cast` by emulating it on int64 lanes with portable saturation; added Einsum ellipsis expansion with NumPy-style broadcasting; allowed bool in `Flatten`, `Gather`, `Concat`, `Slice`, `Transpose`, and `Unsqueeze`; and allowed an empty shape vector in CUDA `ConstantOfShape` for scalar output. Unit tests cover Cast int32 in the kernel, shape, and loader paths; Einsum ellipsis with batch matmul, broadcast, malformed and implicit-negative cases; Flatten bool on CPU and CUDA; ConstantOfShape with an empty shape; and Einsum CUDA with shared ellipsis expansion and double accumulation. OFF build 12/12; CUDA shape 3/3, misc 4/4, index 1/1, concat 2/2; format clean. No oracle-divergent guards were added. Saturating int32 is documented against x86 ONNX Runtime outside the int32 range; in-range results are bit-identical. |
| `V-3b` | OWLv2 CUDA gate: full 1219-node dispatch in `nert-gpu:26.08` (about 7 s, versus roughly one hour on CPU) against ONNX Runtime 1.27.0 | Pass | 2026-10-18 | Logits are bit-identical (top-1 `3600/3600`); predicted boxes differ by at most `1.898e-04`; no row exceeds `1e-3 · (1 + |reference|)`. The bring-up set and a CUDA `Einsum` row closed the remaining gap; no tolerance deviation was needed. |
| `V-4` | First measured coverage pass: 24 local ONNX exports inventoried; kernel tables compared (CPU 72, CUDA 72, OpenCL and HIP 9 at the time); per-model missing rows and gates recorded in [`coverage.md`](coverage.md) | Pass, first pass | 2026-10-18 | CUDA had zero missing operator types for all five loadable models at that point. Nineteen of the 24 exports were blocked by opset. Portable-device gaps and non-kernel blockers were listed. The counts in this row are historical; the live matrix is authoritative and should be updated as rows and gates land. |
