# Provenance — EdgeCrafter ecpose_s opset-18 ONNX artifact (entry 21)

Packet: export-first provisioning for catalog entry 21 (EdgeCrafter).
Produced: 2026-10-23
Deliverable: `/tmp/opencode/exports/ecpose_s/ecpose_s.onnx` (+ `ecpose_s.onnx.data`, external weights)

## Source (read-only, never modified)

- Upstream artifact: `/home/oli/repos/edgecrafter-cpp-inference/models/ecpose_s.onnx`
  (+ `ecpose_s.onnx.data`)
- Method: **no re-export**. The local opset-18 artifacts were treated as the
  provisioned export; they were byte-copied into the writable directory.
- Exact provisioning commands (no re-export):
  ```bash
  mkdir -p /tmp/opencode/exports/ecpose_s
  cp /home/oli/repos/edgecrafter-cpp-inference/models/ecpose_s.onnx /tmp/opencode/exports/ecpose_s/
  cp /home/oli/repos/edgecrafter-cpp-inference/models/ecpose_s.onnx.data /tmp/opencode/exports/ecpose_s/
  ```

## Hashes (source and working copy identical)

- `ecpose_s.onnx` sha256: `810dcac9622028160b97a24b0ae78c6f673350ac131285814acc7bfb37c5c33a` (41834767 bytes)
- `ecpose_s.onnx.data` sha256: `60669d0a32e5f121bb37fda1d3fdb65a150fc96dfe24d31ca9aeb99597a7f388` (41013000 bytes)

## Emitted graph facts (declared by the graph, not assumed)

- `ir_version`: 10
- `opset_import`: `[('', 18)]` — the ai.onnx opset is exactly **18**
- Node count (on-disk): **1534**
- Initializers: 479
- Graph inputs (declared): `images` [N,3,640,640] float32; `orig_target_sizes` [N,2] int64 (symbolic batch N)
- Graph outputs (declared): `scores` [N,60] float32; `labels` [N,60] int64; `keypoints` [N,60,17,2] float32
- `onnx.checker.check_model`: **passes** (see `verify_export.py`)
- External data file: `ecpose_s.onnx.data` alongside the model (loader resolves it by basename)

SYMBOLIC batch dim N on inputs and outputs. Gate fixture freezes N=1 (nert_gate resolves dynamic dims to 1; deterministic 640x640 fixture with orig_target_sizes [[640,640]]).
## Toolchain (inspection only, no export run)

- onnx 1.22.0, onnxruntime 1.27.0, torch 2.12.0+cu130 (pre-existing environment)
- No pip installs; no exporter executed.

## Scope note (updated 2026-10-24)

The original packet provisioned only the export. The release-gate packet has
now implemented the required Split/Sign support and run CPU/CUDA/OpenCL full
model checks. See ../results.md and the per-device raw/task metrics: these
distinguish strict raw parity from the approved fixture-specific detection
criterion. No model or external weight bytes were changed.
