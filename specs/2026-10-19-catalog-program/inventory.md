# Neuriplo catalog inventory

Date: 2026-10-20. Release scope: **30 in-scope ONNX entries** out of 32
catalog groups. NERT is ONNX-only; image understanding (GGUF/llama.cpp) and
Gaussian Splatting are approved out of scope. The full census is retained at
`$HERMES_CACHE/cache/scratch/nert-specs-slim-2026-10-19/inventory-full.md`.

## Release scoreboard

| Status | Count | Entries |
| --- | ---: | --- |
| PASS on CPU/CUDA/OpenCL | 15 | ResNet-18, YOLO26n, OWLv2, RF-DETR seg-small, RAFT-large, YOLO11s, YOLOv8n-pose, YOLO26s-pose, RF-DETR nano-1101, YOLO26n-depth, RF-DETR keypoint-preview, YOLO26s\*, YOLO26x\*, RT-DETRv4-s, RF-DETR seg-nano-576\* |
| Open | 1 | Grounding DINO (OpenCL backend gap) |
| Missing ONNX export | 14 | See export plan below |
| Out of scope | 2 | Image understanding, Gaussian Splatting |
| **Total** | **32** | |

\* YOLO26s and YOLO26x carry the documented R-6 TopK tie-flip deviation in the
sub-0.001 confidence tail; confident detections are identical. RF-DETR
seg-nano-576 carries the documented rank-99 TopK near-tie deviation (`SN-DEV`).

## Entry status

| # | Entry | Status | Blocker / note |
| ---: | --- | --- | --- |
| 1 | ResNet-18 | PASS | Pin the exact fixture: the `/tmp` export uses `GlobalAveragePool`; the Neuriplo parity fixture uses `ReduceMean`. |
| 2 | YOLO26n | PASS | OpenCL gate closed. |
| 3 | OWLv2-base-patch16 | PASS | — |
| 4 | RF-DETR seg-small | PASS | — |
| 5 | RAFT-large | PASS | Real torchvision opset-18 export. |
| 6 | Grounding DINO | Open | CPU/CUDA PASS under the task-equivalence deviation (decoded-set check closed 2026-10-20); OpenCL is a backend GAP after the arena-view revert. |
| 7 | RF-DETR keypoint-preview | PASS | — |
| 8 | RF-DETR seg-nano-576 | PASS\* | Passes with the documented rank-99 TopK near-tie deviation (`SN-DEV`): a 1.43e-06 tie against 4.82e-05 of fp32 reduction-order noise, not a kernel defect. |
| 9 | RF-DETR nano-1101 | PASS | — |
| 10 | YOLO26s | PASS\* | R-6 sub-0.001 tie deviation. |
| 11 | YOLO26x | PASS\* | R-6 sub-0.001 tie deviation. |
| 12 | YOLO26n-depth | PASS | — |
| 13 | YOLOv8n-pose | PASS | — |
| 14 | YOLO11s | PASS | — |
| 15 | YOLO26s-pose | PASS | — |
| 16 | RT-DETRv4-s | PASS | CUDA asymmetric-pad Conv remedy accepted; full-model CUDA gate closed (boxes 0/1200, scores 0/300, labels 300/300). |
| 17 | YOLO detector v5–v12 variants | Missing export | Some checkpoints local; v5/v6/v7 and others need downloads. |
| 18 | YOLO segmentation variants | Missing export | `yolo11m-seg.pt` local; other sizes need downloads. |
| 19 | EdgeCrafter detection | Missing export | Upstream repo + checkpoint download. |
| 20 | EdgeCrafter segmentation | Missing export | Upstream repo + checkpoint download. |
| 21 | EdgeCrafter pose | Missing export | Upstream repo + checkpoint download. |
| 22 | RF-DETR detection other sizes | Missing export | `rfdetr` import broken under transformers 4.35; weights auto-download. |
| 23 | RF-DETR segmentation other sizes | Missing export | Same dependency issue. |
| 24 | RF-DETR keypoint other sizes | Missing export | Same; exporter may fall back to keypoint-preview. |
| 25 | OWL-ViT | Missing export | No exporter script in the repo; needs new code + checkpoint. |
| 26 | ViTPose-base-simple | Missing export | Exporter exists; HF download and likely transformers upgrade. |
| 27 | Depth-Anything-V2 | Missing export | Exporter exists; HF download. |
| 28 | LGM/LGM-mini/GRM | Out of scope | Gaussian rendering/rasterisation outside ONNX; no viable ONNX gate. |
| 29 | VideoMAE/ViViT/TimeSformer | Missing export | Exporters exist; HF downloads. |
| 30 | Classification ViT/TF/other torchvision | Missing export | Exporters exist; downloads; pin opset 18. |
| 31 | RAFT-small | Missing export | Weights download; exporter hardcodes opset 16, convert to 18. |
| 32 | Image understanding (Gemma 4/Llama) | Out of scope | GGUF + llama.cpp; no ONNX runtime contract. |

## Export plan

| Group | Feasibility | Main dependency |
| --- | --- | --- |
| YOLO detector variants | Offline for yolo26s/x/11s/26n; download for v5/v6/v7/v8/v9/v10/v12 | `ultralytics`, checkpoints, opset 18 pin |
| YOLO segmentation variants | Offline for yolo11m-seg; download for other sizes | Ultralytics CLI, `-seg.pt` checkpoints |
| EdgeCrafter detection/seg/pose | Download + new environment | Upstream repo, checkpoints, `onnxsim`/`onnxscript` |
| RF-DETR other sizes | Download + dependency fix | `rfdetr`, transformers upgrade, weights |
| OWL-ViT | New exporter + download | HF `google/owlvit-*` checkpoint |
| ViTPose | Download | HF `usyd-community/vitpose-base-simple`, transformers |
| Depth-Anything-V2 | Download | HF `depth-anything/Depth-Anything-V2-*` |
| VideoMAE / ViViT / TimeSformer | Download | HF checkpoints; 5D input fixtures |
| Classification non-ResNet | Download | Torchvision/timm or TensorFlow weights |
| RAFT-small | Download + conversion | Torchvision weights; 16→18 conversion |

Every new export needs an explicit opset-18 result and a load/probe before
kernel work. Exporter defaults are older: YOLO 12, RF-DETR/OWLv2/Grounding
DINO 17, ViTPose/Depth/video 14, RAFT/RT-DETR 16, ViT classifier 17.

## Gate requirements

- A model is PASS only with a full-model run against the reference on CPU,
  CUDA and OpenCL, reviewer PASS, and a commit ID.
- Op-table presence is OPS, not correctness. The current OpenCL table has 68
  rows; the four absent types (`Abs`, `Floor`, `GreaterOrEqual`, `Neg`) are
  unused by currently loadable entries.
- HIP is unsupported for 0.2.0: the table has nine rows and the host has no
  ROCm toolchain or AMD device.
- Host postprocessing (tokenizer, NMS, mask/flow resize) is outside the ONNX
  gate and must be recorded separately.
- No release while any in-scope entry is Open or Missing-export.

## Known release risks

1. **Fixture pinning.** ResNet-18 PASS depends on which 49-node export is
   gated; do not conflate the `/tmp` export with the Neuriplo parity fixture.
2. **Opset drift.** NERT accepts exactly opset 18; `onnx.checker` does not
   prove semantic equivalence after conversion.
3. **Postprocessing gaps.** OWLv2/Grounding DINO need host tokenization;
   detection, segmentation, depth and optical-flow postprocessors are not
   covered by the ONNX gate.
4. **Consumer device selection.** Neuriplo selects only CPU or opt-in CUDA
   for NERT; OpenCL and HIP are engine capabilities, not consumer-release
   capabilities.
5. **Dynamic shapes.** RF-DETR concatenates a zero-length slice; RAFT, video
   and Gaussian exports need explicit static-shape fixtures for repeatable
   gates.
