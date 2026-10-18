# Validation — pending full-model gates

Defined before running. Bars: RF-DETR detection-identical to the ORT
reference (R-6 precedent); RAFT runs only with a real export.

## Checks

- [V-G1] RF-DETR seg-small full CUDA gate on the 1255-node graph
  vs the ORT reference: boxes, labels top-1, binarized mask IoU.
- [V-G2] Root-cause and fix any CUDA divergence; rerun without env
  overrides.
- [V-G3] Zero-extent Concat regression (CPU/CUDA/OpenCL).
- [V-G4] RAFT export identity check (historical).
- [V-G7] Real torchvision RAFT-large opset-18 export + full CUDA gate.
- [V-G5] OFF/CUDA/OpenCL/cuDNN legs green; format clean.
- [V-G6] Diff audit: packet + gate-fix paths only.

## Acceptance

```bash
cmake --build build && ctest --test-dir build --output-on-failure
cmake --build build-cuda && ctest --test-dir build-cuda --output-on-failure
cmake --build build-ocl && ctest --test-dir build-ocl --output-on-failure
cmake --build build-cudnn && LD_LIBRARY_PATH=$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib:/tmp/opencode/cudnn/lib64 ctest --test-dir build-cudnn --output-on-failure
./scripts/format.sh --check
```

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-G1 | RF-DETR seg-small CUDA gate (1255 nodes, 2 s) vs ORT 1.27.0 reference: boxes maxdiff 9.938e-04, labels maxdiff 3.080e-04, top-1 100%, binarized mask IoU min 0.999528 | pass | 2026-10-18 | CPU control on the same tree: boxes 5.155e-04, labels 4.025e-04, top-1 100%, IoU 0.999669. Detection-identical. |
| V-G2 | TF32 root cause: `NVIDIA_TF32_OVERRIDE=0` reproduced the passing numbers; `CUBLAS_PEDANTIC_MATH` + `CUDNN_FMA_MATH` then did so without the env | pass | 2026-10-18 | Before: boxes 5.818e-01, labels 1.786e+00, mask IoU 0.0; 12 TopK index swaps among near-ties. Blocker chain: TF32 → score noise ~1e-4 → TopK order → slot-indexed decoder positions. |
| V-G3 | Zero-extent Concat: `[2,0,4] + [2,3,4]` axis 1 on CPU/CUDA/OpenCL | pass 3/3 | 2026-10-18 | Fix `329c2e7`; CUDA/OpenCL validation now `dim < 0`. |
| V-G4 | RAFT export identity: `ref.onnx` inputs A/B [B,6,160,160] → trans/rot [1,3], ops include no GridSample/GRU/correlation | resolved by V-G7 | 2026-10-18 | Not RAFT; matrix row corrected; real export provisioned 2026-10-19. |
| V-G7 | RAFT-large (torchvision 0.27, torch 2.12) opset-18 export, full CPU+CUDA gate on `bus.jpg` + 4 px shift [1,3,368,496] vs ORT 1.27.0 | pass | 2026-10-19 | 1,836 ONNX nodes (1,299 folded); CPU maxdiff 1.317e-04 (1,074 s), CUDA 1.344e-04 (305 s), mean ~1.1e-05, tolerance 5.324e-03; CPU-vs-CUDA 1.040e-04. |
| V-G5 | OFF 12/12, CUDA-ON 24/24, OpenCL-ON 16/16, cuDNN farm 24/24, format clean | pass | 2026-10-19 | OpenCL count includes MatMul + LayerNormalization. |
| V-G6 | Diff audit | pass | 2026-10-18 | `329c2e7` review fixes + this packet + TF32 math mode. |

## Deviations

- RAFT export/gate artifacts live under `/tmp/opencode/raft/` and are
  not committed; the export command and inventory are in the packet.
- RAFT CPU cross-check passed (1.317e-04, 1,074 s); CPU-vs-CUDA 1.040e-04.
- RF-DETR gate ran on the throwaway runner in `/tmp/opencode` (not
  committed), like the CPU bring-up; the graph and reference are
  external artifacts.
