# Validation — Other task models on NERT

Defined before implementation. Bars: classification matches top-1 with
score diff at or under 1e-4 on CPU; GPU deviations follow the R-6
precedent (measured, detection-identical, maintainer-noted).

## Checks

- [V-1] ResNet-18 on CUDA: top-1 matches ORT, score diff recorded.
- [V-2] DETR: export inventoried (node/op list on file), GPU gaps
  enumerated as slice entries.
- [V-3] OWLv2 / Grounding-DINO: CPU parity numbers recorded; GPU gaps
  enumerated as slice entries.
- [V-4] Coverage matrix exists and is current through these phases.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | ResNet-18 CUDA gate (nert-gpu:26.08, cuDNN) vs ORT 1.27.0: top-1 107 = 107, top-5 identical order, max abs diff 2.167e-03 | pass w/ deviation | 2026-10-17 | 1094ms CUDA; all 49 nodes dispatched (run-to-completion proof). Deviation: 20 cuDNN convs compound float accum vs ORT CPU kernels; classification-identical like R-6. Export: torchvision 0.27, opset 18, seed 0. |
| T1a | GlobalAveragePool CPU+CUDA rows (exact channel-means, rank-1 negative, shape collapse, loader allowlist) + OFF 8/8 + host CUDA GAP parity + format | pass | 2026-10-17 | Surprise gap from the 49-node inventory (only missing op); bit-exact double accum both sides. |
| — | — | pending | — | — |
