# Plan — opset-18 conversion sweep

Single phase. Convert into `/tmp/opencode/opset18/` (external artifacts,
not committed), verify with `onnx.checker`, then probe with
`LoadGraphFromFile` + `Model` construction on CPU (shape inference
only). Record op-type gaps against the four kernel tables from the
converted ONNX (Constant excluded; the loader folds it).

Fix the loader only when the converted graph is blocked: INT32
constants need int64-lane storage, and the shape-inference constant
evaluator needs `Mod`, `Transpose`, and one-axis N-D `Slice` for the
Grounding DINO pad chain. Re-run all four test legs after the shared
code changes.

## Validation

Evidence in `validation.md`.
