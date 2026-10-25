# Provenance — EdgeCrafter ecseg_s opset-18 ONNX artifact (entry 20)

Packet: export-first provisioning for catalog entry 20 (EdgeCrafter).
Produced: 2026-10-23
Deliverable: `/tmp/opencode/exports/ecseg_s/ecseg_s.onnx` (+ `ecseg_s.onnx.data`, external weights)

## Source (read-only, never modified)

- Upstream artifact: `/home/oli/repos/edgecrafter-cpp-inference/models/ecseg_s.onnx`
  (+ `ecseg_s.onnx.data`)
- Method: **no re-export**. The local opset-18 artifacts were treated as the
  provisioned export; they were byte-copied into the writable directory.
- Exact provisioning commands (no re-export):
  ```bash
  mkdir -p /tmp/opencode/exports/ecseg_s
  cp /home/oli/repos/edgecrafter-cpp-inference/models/ecseg_s.onnx /tmp/opencode/exports/ecseg_s/
  cp /home/oli/repos/edgecrafter-cpp-inference/models/ecseg_s.onnx.data /tmp/opencode/exports/ecseg_s/
  ```

## Hashes (source and working copy identical)

- `ecseg_s.onnx` sha256: `6dc6027f82f297ef36995490c59c2a0c7270aed47c8ca5543c1d38f333dbd179` (42320129 bytes)
- `ecseg_s.onnx.data` sha256: `0c3974d3094377427ddb1fa3707d697073b2e4e9e78fc8d756400e5efd5c1431` (41471792 bytes)

## Emitted graph facts (declared by the graph, not assumed)

- `ir_version`: 10
- `opset_import`: `[('', 18)]` — the ai.onnx opset is exactly **18**
- Node count (on-disk): **1569**
- Initializers: 497
- Graph inputs (declared): `images` [N,3,640,640] float32; `orig_target_sizes` [N,2] int64 (symbolic batch N)
- Graph outputs (declared): `labels` [1,300] int64; `boxes` [1,300,4] float32; `scores` [1,300] float32; `masks` [1,300,160,160] float32
- `onnx.checker.check_model`: **passes** (see `verify_export.py`)
- External data file: `ecseg_s.onnx.data` alongside the model (loader resolves it by basename)

SYMBOLIC batch dim N on both inputs (outputs are static [1,...]). Gate fixture freezes N=1 (nert_gate resolves dynamic dims to 1; deterministic 640x640 fixture with orig_target_sizes [[640,640]]).
## Toolchain (inspection only, no export run)

- onnx 1.22.0, onnxruntime 1.27.0, torch 2.12.0+cu130 (pre-existing environment)
- No pip installs; no exporter executed.

## Scope note (updated 2026-10-24)

The original packet provisioned only the export. The release-gate packet has
now implemented the required Split/Sign support and run CPU/CUDA/OpenCL full
model checks. See ../results.md and the per-device raw/task metrics: these
distinguish strict raw parity from the approved fixture-specific detection
criterion. No model or external weight bytes were changed.
