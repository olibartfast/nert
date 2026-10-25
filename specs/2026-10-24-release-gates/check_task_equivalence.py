#!/usr/bin/env python3
"""Measure proposed task-equivalence acceptance; this does not waive raw failures.

python3 .../check_task_equivalence.py /tmp/nert-release-gates MODEL DEVICE
Confidence is fixed at 0.25. References must be from the exact gated artifact.
"""
import json
import sys
from pathlib import Path

import numpy as np

root, model, device = sys.argv[1:]
path = Path(root) / model
refs = {p.stem[4:]: np.load(p, allow_pickle=False) for p in path.glob("ref_*.npy")}
if not refs:
    raise SystemExit("no references")
got = {}
for name, ref in refs.items():
    output = path / (device + "_" + name + ".bin")
    if output.stat().st_size != ref.nbytes:
        raise SystemExit("output byte size mismatch: " + name)
    got[name] = np.fromfile(output, dtype=ref.dtype).reshape(ref.shape)
    if not np.isfinite(ref).all() or not np.isfinite(got[name]).all():
        raise SystemExit("non-finite output: " + name)
rf = model.startswith("rfdetr")
if rf:
    ref_labels = refs["labels"].argmax(-1).reshape(-1)
    got_labels = got["labels"].argmax(-1).reshape(-1)
    sigmoid = lambda x: np.exp(-np.logaddexp(0, -x))
    ref_scores = sigmoid(refs["labels"]).max(-1).reshape(-1)
    got_scores = sigmoid(got["labels"]).max(-1).reshape(-1)
else:
    ref_labels = refs["labels"].reshape(-1)
    got_labels = got["labels"].reshape(-1)
    ref_scores = refs["scores"].reshape(-1)
    got_scores = got["scores"].reshape(-1)
keep_ref, keep_got = ref_scores >= .25, got_scores >= .25
keep = keep_ref | keep_got
result = {"model": model, "device": device, "threshold": .25,
          "reference_kept": int(keep_ref.sum()), "device_kept": int(keep_got.sum()),
          "selection_equal": bool(np.array_equal(keep_ref, keep_got)),
          "kept_class_mismatch": int(np.count_nonzero(ref_labels[keep] != got_labels[keep])),
          "reference_threshold_margin": float(np.min(np.abs(ref_scores - .25))),
          "raw_mismatch": {}, "kept_mismatch": {}}
passed = result["selection_equal"] and result["kept_class_mismatch"] == 0 and keep.any()
for name, ref in refs.items():
    actual = got[name]
    if ref.dtype.kind == "f":
        error = np.abs(ref.astype(np.float64) - actual.astype(np.float64))
        bad = error > 1e-3 * (1 + np.abs(ref.astype(np.float64)))
    else:
        bad = ref != actual
    bad = bad.reshape(keep.size, -1)
    result["raw_mismatch"][name] = int(bad.sum())
    result["kept_mismatch"][name] = int(bad[keep].sum())
    passed &= not bad[keep].any()
if "masks" in refs:
    ref_mask = (refs["masks"] >= 0).reshape(keep.size, -1)
    got_mask = (got["masks"] >= 0).reshape(keep.size, -1)
    union = (ref_mask | got_mask).sum(1)
    intersection = (ref_mask & got_mask).sum(1)
    iou = np.divide(intersection, union, out=np.ones(keep.size), where=union > 0)
    result["all_mask_iou_min"] = float(iou.min())
    result["kept_mask_iou_min"] = float(iou[keep].min()) if keep.any() else None
    result["kept_binary_mask_mismatch"] = int(np.count_nonzero(ref_mask[keep] != got_mask[keep]))
    passed &= result["kept_binary_mask_mismatch"] == 0
result["task_equivalence_pass"] = bool(passed)
print(json.dumps(result, indent=2))
sys.exit(0 if passed else 1)
