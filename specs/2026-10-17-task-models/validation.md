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
| V-2 | RF-DETR seg-small inventory (1784 nodes, opset 17→18 bit-exact conversion): CPU covers all non-folded ops; CUDA gaps filed R1..R6b | pass | 2026-10-17 | Full CPU run 571s exit 0 (all nodes dispatched). CPU-vs-ORT: boxes 5.2e-04, labels top-1 100%, masks binarized IoU min 1.000000 — model-identical. Slices: R1 LayerNormalization (56), R2 Sqrt (48), R3 Erf (17), R4 ReduceSum (5), R5 GridSample (4), R6a Sin/Cos/Exp (11), R6b Tile/Not/Greater/Less/And (9). Einsum stays CPU-only. |
| — | — | pending | — | — |
| V-3 | OWLv2-base-patch16 CPU parity (1219 nodes, opset 18 native, NERT-side weights — answers Q-2 for open-vocab): logits maxdiff 0 (top-1 3600/3600), pred_boxes maxdiff 1.297e-04, 0/3600 over tolerance; 3037 s Release | pass | 2026-10-18 | CUDA gaps overlap RF-DETR slices (LayerNorm R1, Erf R3) plus O1 IsNaN (12), O2 ReduceL2 (3), O3 And/LessOrEqual/ArgMax/Greater/Elu (6). Einsum stays CPU-only. |
| V-3a | OWLv2 CPU bring-up fixes found by the real model: Cast to=6 (INT32) emulated on int64 lanes with portable saturation; Einsum ellipsis expansion (shared loader/shape/kernel home, NumPy broadcast); Flatten accepts bool (text mask); CUDA ConstantOfShape accepts an empty shape vector (scalar output); Gather accepts bool data (text mask), both matching the CPU oracle. Unit tests: Cast int32 (kernel/shape/loader), Einsum ellipsis (batch matmul, broadcast, malformed/implicit negatives), Flatten bool (CPU + CUDA), ConstantOfShape empty-shape, Gather/Cast int32 the bool movement family Concat/Slice/Transpose/Unsqueeze, and Einsum CUDA (shared ellipsis expansion, double accumulation; CPU + CUDA, oracle-compared); OFF 12/12, CUDA shape 3/3 + misc 4/4 + index 1/1 + concat 2/2, format | pass | 2026-10-18 | No oracle-divergent guards; saturating int32 is documented vs x86 ORT past range, in-range bit-identical. |
| V-3b | OWLv2 CUDA gate: full 1219-node dispatch in `nert-gpu:26.08` (7 s vs CPU ~1 h) against the ORT 1.27.0 reference: logits maxdiff 0 (top-1 3600/3600), pred_boxes maxdiff 1.898e-04, 0/3600 over `1e-3·(1+|ref|)` | pass | 2026-10-18 | Closed by the bring-up set plus a CUDA Einsum row (shared ellipsis expansion, double accumulation); no tolerance deviation. |
