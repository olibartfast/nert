# Validation — opset-18 conversion sweep

Defined before conversion. Bars: converted graph passes
`onnx.checker`; NERT loads it and infers shapes with concrete inputs;
all existing test legs stay green; gaps recorded per device.

## Checks

- [V-C1] Convert every local export outside opset 18; checker passes.
- [V-C2] NERT load + shape inference per converted graph; record
  blockers for failures.
- [V-C3] Loader fixes (INT32 constants, `Mod`/`Transpose`/N-D `Slice`
  evaluation) with all test legs green.
- [V-C4] Per-model op gaps vs CPU/CUDA/OpenCL/HIP tables.
- [V-C5] OFF/CUDA/OpenCL/cuDNN legs green; format clean.
- [V-C6] Diff audit: loader + packet + coverage only.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-C1 | `onnx.version_converter` + checker on 17 candidates | pass 17/17 | 2026-10-18 | Outputs in `/tmp/opencode/opset18/`; converted artifacts are external. |
| V-C2 | NERT `LoadGraphFromFile` + shape inference | pass 13/17 | 2026-10-18 | Grounding DINO 19,224 folded nodes now loads. Blocked: FoundationPose `ref2`/`refiner`/`rw`/`sw` and the deep-infer RF-DETR duplicate (dynamic `Slice` parameters). |
| V-C3 | Loader: INT32 `Constant`/`value_info` widened to int64 lanes; evaluator `Mod`, `Transpose`, one-axis N-D `Slice` | pass | 2026-10-18 | Needed by Grounding DINO's constant pad chain (`Shape→Mod→Sub→Unsqueeze→Concat→Cast→Reshape→Slice→Transpose→Reshape→Cast→Pad`). |
| V-C4 | Op gaps from converted ONNX vs tables | pass | 2026-10-18 | CPU/CUDA 0/0 for all 13 loadable models; OpenCL gaps 3–39 types; HIP larger (coverage matrix). |
| V-C5 | OFF 12/12, CUDA-ON 24/24, OpenCL-ON 14/14, cuDNN 24/24, format clean | pass | 2026-10-18 | Shared shape-inference changes re-ran every leg. |
| V-C6 | Diff audit | pass | 2026-10-18 | `src/model_loader.cpp`, `src/shape_inference.cpp`, packet + matrix. |

## Deviations

- No full-model reference gates for the converted variants: no ORT
  references or image fixtures on this host. Recorded as OPS, not PASS.
- `onnx.checker` does not prove semantic equivalence for down-converted
  graphs (19/20→18); only load/shape inference is claimed here.
