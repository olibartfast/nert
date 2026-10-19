# Coverage matrix — task models × devices

Date: 2026-10-20. Release target: 30 in-scope ONNX catalog entries. The
catalog census is [inventory.md](../2026-10-19-catalog-program/inventory.md);
the run history is [ledger.md](../2026-10-19-catalog-program/ledger.md).
Full pre-slim matrix:
`$HERMES_CACHE/cache/scratch/nert-specs-slim-2026-10-19/coverage-full.md`.

## Definitions

- **PASS** — full-model gate recorded against the reference on all three
  listed devices.
- **OPS** — every op type has a row on that device; no full-model gate. This
  is kernel coverage, not correctness.
- **GAP** — at least one op type used by the model has no row on that device.
- **FAIL** — the full-model gate ran and missed tolerance.
- **BLOCKED / MISSING-EXPORT** — no loadable export, so no NERT gate exists.
- **OUT-OF-SCOPE** — no ONNX runtime contract.

Node counts in the detailed packets are NERT post-constant-folding counts.
HIP is not part of the 0.2.0 release gate.

## Kernel tables

| Layer | Rows | Notes |
| --- | ---: | --- |
| CPU (oracle) | 72 | Complete reference set. |
| CUDA | 72 | Complete set; `Conv` and `ConvTranspose` require cuDNN. Strict float32 (`CUBLAS_PEDANTIC_MATH`, `CUDNN_FMA_MATH`). |
| OpenCL | 68 | Complete for every currently loadable catalog entry. Missing `Abs`, `Floor`, `GreaterOrEqual`, `Neg`, none used by those entries. |
| HIP | 9 | Original scaffold only; **unsupported for 0.2.0**. |

## Full-model gates

| Model | Task | CPU | CUDA | OpenCL | Evidence |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 | classification | PASS | PASS | PASS | V-1; OpenCL top-1 107 = ORT. |
| YOLO26n | detection | PASS | PASS | PASS | R-6; OpenCL maxdiff 4.88e-04, 0/1800 over tolerance. |
| OWLv2-base-patch16 | open-vocab detection | PASS | PASS | PASS | V-3/V-3b; OpenCL logits exact, top-1 3600/3600. |
| RF-DETR seg-small | detection + segmentation | PASS | PASS | PASS | CUDA mask IoU 0.9995; OpenCL dets 6.82e-04, mask IoU min 0.99967. |
| RAFT-large | optical flow | PASS | PASS | PASS | CPU 1.317e-04, CUDA 1.344e-04, OpenCL 1.323e-04 vs ORT. |
| YOLO11s, YOLOv8n-pose, YOLO26s-pose, RF-DETR nano-1101, YOLO26n-depth, RF-DETR keypoint-preview, YOLO26s, YOLO26x | detection / pose / depth | PASS | PASS | PASS | 2026-10-19 converted-variant gates; YOLO26s/x carry the documented R-6 sub-0.001 TopK deviation. |
| Grounding DINO | open-vocab detection | PASS (GD deviation) | PASS (GD deviation) | GAP | Task-equivalence gate 2026-10-20; gate metrics are post-canonicalization. Raw: pred_boxes 72/3,600 CPU and 80/3,600 CUDA over 1e-3(1+\|ref\|) (max abs 9.74e-01); per-box IoU min 0.0, 882/900 CPU and 880/900 CUDA above 0.999; raw max Δ max-score 2.12e-02 exceeds the 1.39e-02 margin, so "no query can cross" holds only canonicalized. Canonicalized: pred_boxes 0/3600 and per-box IoU min 0.99973 (900/900); top-1 900/900; decoded set_diff 0 at the 0.25/0.25 HF proxy (1 kept query, margin 1.39e-02, max score delta 5.36e-03); logits envelope diagnostic only, per-device to be recalibrated (canonicalized mean 7.35e-03 CPU, 7.64e-03 CUDA). OpenCL backend GAP: the handle path demands 11,028.5 MiB on a 5,803.3 MiB device (model live set 1,687.9 MiB); rejected remedies — sub-buffer views (measured 70x the arena, driver release crash) and per-view liveness (inherits both); named remedy (per-tensor plain-buffer liveness, or a kernel-offset arena) deferred post-0.2.0. |
| RF-DETR seg-nano-576 | detection + segmentation | PASS | PASS (SN deviation) | PASS (SN deviation) | Rank-99 TopK near-tie (SN-DEV): ORT 182 vs GPU 1503; ORT gap 1.43051e-06 vs max score-vector ORT−device deviation 4.81606e-05 CUDA / 1.24216e-04 OpenCL on `/transformer/ReduceMax_output_0` (2,304 elements). Selection rule already ORT's; no engine change. Unpinned, both GPUs: query 99 flips 64→84, top-1 99/100; dets 50/400; labels 3,629 CUDA / 3,628 OpenCL of 9,100; masks 1,240,141 CUDA / 1,239,727 OpenCL of 2,073,600 over tolerance (max ratio 47,350.3 / 47,350.4). PASS is on the ORT-neutral pin (TopK→2×Constant; ORT max_abs 0): dets 0/400, labels 0/9,100, top-1 100/100 on all three; masks 2/36/17 (CPU/OpenCL/CUDA) of 2,073,600, max ratio ≤ 2.6; IoU min 0.999800 at raw>=0; CUDA 0.984375 at raw>=0.5 (query 36, 63/64; CPU/OpenCL 1.000000). The pinned graph is a single-input counterfactual for this gate (TopK removed; 1,215 vs 1,216 nodes): not a TopK-correctness result, not a deployable substitute. Result-level equivalence, not bit-identity. |
| RT-DETRv4-s | detection | PASS | PASS | PASS | CUDA asymmetric-pad Conv remedy (explicit zero pre-pad in both CUDA Conv rows; symmetric path unchanged); full-model CUDA gate: boxes 0/1200 (max 7.93e-04), scores 0/300 (max 2.46e-05), labels 300/300 exact; 11 detections over 0.25, class ids [5,0,0,0,0,26,9,11,26,26,26]. |
| 14 missing-export groups | all families | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | See inventory; export-first. |
| Image understanding, Gaussian Splatting | — | OUT-OF-SCOPE | OUT-OF-SCOPE | OUT-OF-SCOPE | No ONNX contract. |

## Release scoreboard

| Status | Count | Entries |
| --- | ---: | --- |
| PASS | 15 | Five primary models + eight converted variants + RT-DETRv4-s + RF-DETR seg-nano-576 (SN deviation) |
| Open | 1 | Grounding DINO (CPU/CUDA task-equivalence PASS; OpenCL backend gap) |
| Missing export | 14 | YOLO variants, EdgeCrafter, RF-DETR sizes, OWL-ViT, ViTPose, Depth-Anything-V2, video, classification, RAFT-small |
| Out of scope | 2 | Image understanding, Gaussian Splatting |
| **Total** | **32** | 30 in scope |

15 + 1 + 14 = 30 in-scope entries. The release gate is zero open entries and
zero missing exports among those 30; HIP is outside the gate. The catalog is
not complete and 0.2.0 is not release-ready.

## Open items

Closed and dropped from this list: the Grounding DINO CPU/CUDA
task-equivalence gate, the RF-DETR seg-nano-576 SN-DEV ruling and the
RT-DETRv4-s full-model CUDA gate.

1. **Grounding DINO OpenCL cell** — a recorded backend gap (the handle path
   demands 11,028.5 MiB on a 5,803.3 MiB device). Whether the release gate
   admits a documented GAP-with-deviation is still an open maintainer
   decision.
2. **Missing exports** — provision, probe, gate and review the 14 groups.

## Non-kernel limits

- **Opset policy.** NERT loads exactly opset 18; every new export must be
  pinned or converted, and conversion does not prove semantic equivalence.
- **Postprocessing.** Tokenization, NMS, mask/flow resize and Gaussian
  rendering are host-side and outside the ONNX gate.
- **Consumer adapter.** Neuriplo selects CPU or opt-in CUDA for NERT; there is
  no OpenCL/HIP consumer selection. OpenCL/HIP columns are engine capability.
- **HIP.** Recorded unsupported for 0.2.0: no ROCm toolchain or AMD device on
  the reference host; nine rows only. No paper mirror is planned.

## Sources

- `src/kernels/cpu_kernels.cpp`, `src/cuda/table.cu`, `src/opencl/table.cpp`,
  `src/hip/table.cpp`.
- Full-model evidence: `specs/2026-10-17-task-models/validation.md` and the
  dated validation packets; detailed gate numbers in the pre-slim coverage
  archive and the catalog ledger.
- Catalog program: `specs/2026-10-19-catalog-program/{inventory,ledger}.md`.
