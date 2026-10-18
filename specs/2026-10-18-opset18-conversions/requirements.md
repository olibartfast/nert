# Requirements — opset-18 conversion sweep

Date: 2026-10-18. NERT loads ONNX-domain opset 18 only; 19 of 24
inventoried exports were outside that. Convert them, verify NERT loads
the result, and record per-model device gaps.

## In scope

- `onnx.version_converter.convert_version(model, 18)` for every local
  export outside opset 18, then `onnx.checker.check_model`.
- NERT-side verification: `LoadGraphFromFile` plus shape inference with
  concrete input dims (no execution yet).
- Loader/shape-inference fixes only where a converted graph is blocked
  by a real gap (not by missing weights/exports).

## Out of scope

- Full-model reference gates for the converted variants (no ORT
  references or input fixtures on this host).
- New kernels, portable backends, Neuriplo runs.
- Downloading absent exports (RAFT, DETR/ViT families).

## Findings

- 16 candidates converted and passed `onnx.checker`; a 17th
  (`sw.onnx`) converted too.
- 13 converted graphs load in NERT and are CPU/CUDA op-complete
  (Grounding DINO, two RF-DETR keypoint exports, RF-DETR seg-nano-576,
  RF-DETR nano-1101, FoundationPose scorer, YOLO26s, YOLO26x,
  YOLO26n-depth, YOLO11s, YOLOv8n-pose, YOLO26s-pose). Grounding DINO
  needed two loader fixes: INT32 constants and constant-parameter
  evaluation through `Mod`, `Transpose`, and single-axis N-D `Slice`.
- Remaining conversion blockers: FoundationPose `ref2`/`refiner`/`rw`/
  `sw` and the deep-infer RF-DETR duplicate need dynamic `Slice`
  parameters in shape inference (the NERT-side RF-DETR export already
  loads). RAFT and the DETR/ViT families have no local export.
