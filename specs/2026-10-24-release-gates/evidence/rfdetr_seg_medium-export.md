# Provenance — RF-DETR segmentation-medium opset-18 export (entry 23)

- **Date:** 2026-10-24
- **Model:** RF-DETR SegMedium (`rfdetr` 1.8.0, class `RFDETRSegMedium`)
- **Checkpoint:** `/home/oli/.roboflow/models/rf-detr-seg-medium.pt`
  (pre-cached 2026-08; public Roboflow RF-DETR weights, non-hermetic source;
  no download needed during this session)
- **Export driver:** `RFDETRSegMedium().export(output_dir=...,
  opset_version=18, batch_size=1)` via the venv python
  (the Neuriplo exporter `export/segmentation/rfdetr/export.py` passes a stale
  `simplify=` kwarg rejected by rfdetr 1.8.0's `export()` and was not used)
- **Environment:** isolated venv `/tmp/opencode/venvs/rfdetr`
  (`--system-site-packages` + `transformers==5.20.2`, `torch==2.12.0+cu130`,
  `onnx==1.22.0`); system transformers 4.35.0 cannot import rfdetr
  (`BackboneConfigMixin` needs transformers 5.x)
- **Artifact:** `/tmp/opencode/exports/rfdetr_seg_medium/rfdetr-seg-medium.onnx`
- **sha256:** `8cf435c729191fd5d4b2149c90b5f3a1609bf3c60a252cb274baafc431e19cd0`
- **Bytes:** 137,708,102
- **Opset:** pinned at export (`opset_version=18`); ai.onnx == 18, checker-clean
- **Graph:** 2,128 nodes on disk; input `input` `[1,3,432,432]` float32
  (model default resolution 432 for SegMedium); outputs `dets` `[1,200,4]`,
  `labels` `[1,200,91]`, `masks` `[1,200,108,108]` float32
- **Acceptance:** `verify_export.py` asserts `opset == 18`, checker-clean,
  nodes > 0, non-empty io — 6/6 PASS
- **Caveat:** none on equivalence — opset pinned at export, not converted.
  Weights source is non-hermetic (upstream download); no kernel rows added.
- **Loader note / remedy:** same DINOv2 scalar-Gather Shape subgraph as
  det-small; NERT shape inference rejects Shape-of-scalar — onnxsim remedy
  applied identically: `rfdetr-seg-medium-sim.onnx`, 2,128 → 1,069 nodes,
  opset still 18, checker-clean, io identical, zero `Shape` nodes left.
  Raw-vs-sim ORT on the bus fixture: dets 3/800 over-tol (max 3.7e-3),
  labels 140/18200 (max 5.4e-3), masks 177708/2332800 (max 0.19 at |ref|≤200);
  RAW-vs-RAW+1ulp control shows the same scale (3 / 111 / 113647, masks max
  0.12) — ambient model chaos, onnxsim contributes no more than ~ulp noise.
  Provisioned gate artifact is the `-sim` file; ORT refs are from sim.
- **Gate artifact:** `/tmp/opencode/exports/rfdetr_seg_medium/rfdetr-seg-medium-sim.onnx`
