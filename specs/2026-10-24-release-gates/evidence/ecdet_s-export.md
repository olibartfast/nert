# Provenance — EdgeCrafter ecdet_s opset-18 ONNX artifact (entry 19)

Packet: export-first provisioning for catalog entry 19 (EdgeCrafter).
Produced: 2026-10-23
Deliverable: `/tmp/opencode/exports/ecdet_s/ecdet_s.onnx` (+ `ecdet_s.onnx.data`, external weights)

## Source (read-only, never modified)

- Upstream artifact: `/home/oli/repos/edgecrafter-cpp-inference/models/ecdet_s.onnx`
  (+ `ecdet_s.onnx.data`)
- Method: **no re-export**. The local opset-18 artifacts were treated as the
  provisioned export; they were byte-copied into the writable directory.
- Exact provisioning commands (no re-export):
  ```bash
  mkdir -p /tmp/opencode/exports/ecdet_s
  cp /home/oli/repos/edgecrafter-cpp-inference/models/ecdet_s.onnx /tmp/opencode/exports/ecdet_s/
  cp /home/oli/repos/edgecrafter-cpp-inference/models/ecdet_s.onnx.data /tmp/opencode/exports/ecdet_s/
  ```

## Hashes (source and working copy identical)

- `ecdet_s.onnx` sha256: `ff47e8f1e9b1d14bf8a4849faa74e9ae12cc10bad1042f4b6e4d359acb78437e` (40674706 bytes)
- `ecdet_s.onnx.data` sha256: `daad7609070118daf50fe8e427f97c79bcdd185365d3c5c5200de8278d4d9706` (39832880 bytes)

## Emitted graph facts (declared by the graph, not assumed)

- `ir_version`: 10
- `opset_import`: `[('', 18)]` — the ai.onnx opset is exactly **18**
- Node count (on-disk): **1523**
- Initializers: 472
- Graph inputs (declared): `images` [1,3,640,640] float32; `orig_target_sizes` [1,2] int64
- Graph outputs (declared): `labels` [1,300] int64; `boxes` [1,300,4] float32; `scores` [1,300] float32
- `onnx.checker.check_model`: **passes** (see `verify_export.py`)
- External data file: `ecdet_s.onnx.data` alongside the model (loader resolves it by basename)

STATIC batch: batch dim is literal 1 in all inputs/outputs; no symbolic dims.
## Toolchain (inspection only, no export run)

- onnx 1.22.0, onnxruntime 1.27.0, torch 2.12.0+cu130 (pre-existing environment)
- No pip installs; no exporter executed.

## Scope note (updated 2026-10-24)

The original packet provisioned only the export. The release-gate packet has
now implemented the required Split/Sign support and run CPU/CUDA/OpenCL full
model checks. See ../results.md and the per-device raw/task metrics: these
distinguish strict raw parity from the approved fixture-specific detection
criterion. No model or external weight bytes were changed.
