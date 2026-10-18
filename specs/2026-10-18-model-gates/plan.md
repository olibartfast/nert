# Plan — pending full-model gates

Single phase. Use the existing throwaway runners (outside the tree):
RF-DETR on `/tmp/opencode/rfdetr-seg-small-18.onnx` with
`/tmp/opencode/rfdetr_input.raw`, CPU reference from the recorded gate
(171 s on the current tree), ORT reference
`/tmp/opencode/rfdetr_ort18.npy` (dets 400, labels 9100, masks
921600). cuDNN leg built with `CUDNN_ROOT=/tmp/opencode/cudnn`.

Bars: detection-identical per the R-6 precedent (boxes within the
recorded CPU-vs-ORT scale, class top-1 identical, binarized mask IoU
effectively 1). Any blocker fixed with a regression test and all legs
re-run. RAFT runs only if a real opset-18 export is found.

## Validation

Evidence in `validation.md`.
