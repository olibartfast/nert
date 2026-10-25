# Provenance — RF-DETR detection-small opset-18 export (entry 22)

- **Date:** 2026-10-24
- **Model:** RF-DETR Small detection (`rfdetr` 1.8.0, class `RFDETRSmall`)
- **Checkpoint:** `/home/oli/.roboflow/models/rf-detr-small.pth` (368,090,xxx bytes;
  downloaded by the rfdetr package from `storage.googleapis.com`, MD5-validated by
  the package at download; public Roboflow RF-DETR weights, non-hermetic source)
- **Export driver:** `/tmp/opencode/exports/rfdetr_det_small/export_small18.py`
  (calls `RFDETRSmall().export(output_dir=..., opset_version=18, batch_size=1)` directly;
  the Neuriplo exporter `export/detection/rfdetr/export.py` passes a stale
  `simplify=` kwarg rejected by rfdetr 1.8.0's `export()` and was not used)
- **Environment:** isolated venv `/tmp/opencode/venvs/rfdetr`
  (`--system-site-packages` + `transformers==5.20.2`, `torch==2.12.0+cu130`,
  `onnx==1.22.0`); system transformers 4.35.0 cannot import rfdetr
  (`BackboneConfigMixin` needs transformers 5.x)
- **Command:** `/tmp/opencode/venvs/rfdetr/bin/python export_small18.py small
  /tmp/opencode/exports/rfdetr_det_small`
- **Artifact:** `/tmp/opencode/exports/rfdetr_det_small/rfdetr-small.onnx`
- **sha256:** `bb8588c9b36402a8c2504bce0861ac8ef3abba68b35698fecdd5db6bb2c134ad`
- **Bytes:** 120,815,153
- **Opset:** pinned at export (`opset_version=18`); ai.onnx == 18, checker-clean
- **Graph:** 1,712 nodes on disk; input `input` `[1,3,512,512]` float32
  (model default resolution 512 for Small); outputs `dets` `[1,300,4]`,
  `labels` `[1,300,91]` float32
- **Acceptance:** `verify_export.py` asserts `opset == 18`, checker-clean,
  nodes > 0, non-empty io — 6/6 PASS (against the raw export)
- **Loader remedy (onnxsim simplify, documented exporter option):** NERT's
  static shape inference (`src/shape_inference.cpp:2302-2314`, `shape_shape`)
  rejects `Shape` of a rank-0 tensor with "Shape selects no dims", while NERT's
  own constant evaluator (same file:708-714) and ORT accept it. The DINOv2
  backbone in this size feeds `/transformer/Shape` from a scalar int64 Gather
  chain (ORT-probed runtime value: scalar 0 → Shape()=empty → Expand=scalar 32;
  the whole ScatterND→Gather→Gather_1→Shape→Expand_4 DAG is constant-rooted).
  `onnxsim simplify` folds it away: `rfdetr-small-sim.onnx`, 1,712 → 844 nodes,
  opset still 18, checker-clean, io identical, zero `Shape` nodes left.
  Raw-vs-sim ORT equivalence: bit-identical NO (fp reassociation); on 3 stable
  inputs max_abs ≤ 9.4e-05 (0 over 1e-3-scaled-tol); on 1 chaotic input both
  raw-vs-sim AND raw-vs-raw+1ulp diverge identically (dets 0.91/labels 1.43) —
  the model is intrinsically ill-conditioned there, onnxsim contributes no more
  than 1 ulp of input noise. Provisioned gate artifact is the `-sim` file;
  refs below are ORT(sim).
- **Gate artifact:** `/tmp/opencode/exports/rfdetr_det_small/rfdetr-small-sim.onnx`
  (1712 raw → 844 dispatched; node deltas are folded Constants)
- **Caveat:** none on equivalence — opset pinned at export, not converted.
  Weights source is non-hermetic (upstream download); no kernel rows added.
