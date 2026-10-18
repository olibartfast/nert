# Requirements — pending full-model gates (RF-DETR, RAFT)

Date: 2026-10-18. Follow-up to the task-models coverage matrix: RF-DETR
seg-small and RAFT had complete CUDA op coverage but no full-model gate.

## In scope

- Run the RF-DETR seg-small CUDA gate against the ORT reference.
- Provision the real torchvision RAFT-large export and run the CUDA
  gate against ORT.
- Fix only blockers the gates surface; no speculative kernel work.

## Out of scope

- New CUDA kernels (coverage is complete), portable backends, Neuriplo
  adapter runs, bench/.

## Findings

- **RF-DETR exposed two blockers.** (1) `Concat` rejected zero-extent
  operands; the transformer concatenates a zero-length slice. Fixed in
  `329c2e7` across CPU/CUDA/OpenCL with regression tests. (2)
  `CUBLAS_DEFAULT_MATH`/`CUDNN_DEFAULT_MATH` still permit TF32 on
  Ampere+, contrary to the code comments; TF32 noise flipped 12
  near-tie TopK selections, which changed decoder slot assignment and
  the outputs by O(0.5). Fixed here with `CUBLAS_PEDANTIC_MATH` and
  `CUDNN_FMA_MATH`.
- **RAFT was misidentified, then provisioned.** The local `ref.onnx`
  is a pose-registration refiner (inputs `A`/`B` 6×160×160, outputs
  `trans`/`rot`), not torchvision RAFT (which outputs flow and uses
  correlation/GRU ops). A real torchvision RAFT-large export was then
  produced at opset 18 (1,836 ONNX nodes, 1,299 folded) and gated:
  CUDA-vs-ORT flow maxdiff 1.344e-04 (tolerance 5.324e-03) on a real
  `bus.jpg` pair with a 4 px shift.

## Open questions

- RAFT CPU cross-check on the new export is running; the CUDA gate is
  already against ORT on the same graph.
