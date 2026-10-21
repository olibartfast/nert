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

The CUDA 72 rows assume a cuDNN-enabled build: `Conv` and `ConvTranspose` sit
behind `#ifdef NERT_HAS_CUDNN`, so a non-cuDNN CUDA build would not have those
rows.

## Full-model gates

| Model | Task | CPU | CUDA | OpenCL | Evidence |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 | classification | PASS | PASS | PASS | V-1; OpenCL top-1 107 = ORT. |
| YOLO26n | detection | PASS | PASS | PASS | R-6; OpenCL maxdiff 4.88e-04, 0/1800 over tolerance. |
| OWLv2-base-patch16 | open-vocab detection | PASS | PASS | PASS | V-3/V-3b; OpenCL logits exact, top-1 3600/3600. |
| RF-DETR seg-small | detection + segmentation | PASS | PASS | PASS | CUDA mask IoU 0.9995; OpenCL dets 6.82e-04, mask IoU min 0.99967. |
| RAFT-large | optical flow | PASS | PASS | PASS | CPU 1.317e-04, CUDA 1.344e-04, OpenCL 1.323e-04 vs ORT. |
| RAFT-small | optical flow | PASS | PASS | PASS | Gate 2026-10-20; opset 18 converted from an opset-16 raw export (conversion does not prove equivalence — the full-model gate is the evidence); 1,630 nodes on disk / **1,115** dispatched equally on all three legs (515 `Constant` folded); `input1`/`input2` [1,3,384,512] float32 → `flow_prediction` [1,2,384,512] float32 (393,216 elements); export sha256 `8be6b770acab2da0093891eb568397b98bb82bb4c54d91298e2a8842f2d7f950`; fixtures `frame1.bin` sha256 `95c3b7f01a346c8306b7c96437b3e8d02deebc1497003e116c83fd1bf80a1e38`, `frame2.bin` sha256 `870533ce8bfac4508b9d2cc2fc35335a7abd798ed4c79c9bc4da29382debf548`; ORT 1.27.0 CPUExecutionProvider reference sha256 `a622b211ac522e5a040df357065c227e1dae0f768a04937e1f18f96547aaf3e5`. `--tol 1e-3` scaled, over_tol 0 on all three legs; max_abs/mean/max_ratio CPU 6.10352e-05/4.84344e-06/0.01218946, CUDA 6.10352e-05/4.85096e-06/0.01224228, OpenCL 6.19888e-05/1.08869e-05/0.01226346. All legs exit 0. Reviewer PASS-WITH-FINDINGS; non-blocking: docs-scoped subject carrying the prior engine fix (hygiene hazard), missing ORT stdout log, float32-vs-float64 mean cosmetic. |
| YOLO11s, YOLOv8n-pose, YOLO26s-pose, RF-DETR nano-1101, YOLO26n-depth, RF-DETR keypoint-preview, YOLO26s, YOLO26x | detection / pose / depth | PASS | PASS | PASS | 2026-10-19 converted-variant gates; YOLO26s/x carry the documented R-6 sub-0.001 TopK deviation. |
| Grounding DINO | open-vocab detection | PASS (GD deviation) | PASS (GD deviation) | GAP | Task-equivalence gate 2026-10-20; gate metrics are post-canonicalization. Raw: pred_boxes 72/3,600 CPU and 80/3,600 CUDA over 1e-3(1+\|ref\|) (max abs 9.74e-01); per-box IoU min 0.0, 882/900 CPU and 880/900 CUDA above 0.999; raw max Δ max-score 2.12e-02 exceeds the 1.39e-02 margin, so "no query can cross" holds only canonicalized. Canonicalized: pred_boxes 0/3600 and per-box IoU min 0.99973 (900/900); top-1 900/900; decoded set_diff 0 at the 0.25/0.25 HF proxy (1 kept query, margin 1.39e-02, max score delta 5.36e-03); logits envelope diagnostic only, per-device to be recalibrated (canonicalized mean 7.35e-03 CPU, 7.64e-03 CUDA). OpenCL backend GAP: the handle path demands 11,028.5 MiB on a 5,803.3 MiB device (model live set 1,687.9 MiB); rejected remedies — sub-buffer views (measured 70x the arena, driver release crash) and per-view liveness (inherits both); named remedy (per-tensor plain-buffer liveness, or a kernel-offset arena) deferred post-0.2.0. |
| RF-DETR seg-nano-576 | detection + segmentation | PASS | PASS (SN deviation) | PASS (SN deviation) | Rank-99 TopK near-tie (SN-DEV): ORT 182 vs GPU 1503; ORT gap 1.43051e-06 vs max score-vector ORT−device deviation 4.81606e-05 CUDA / 1.24216e-04 OpenCL on `/transformer/ReduceMax_output_0` (2,304 elements). Selection rule already ORT's; no engine change. Unpinned, both GPUs: query 99 flips 64→84, top-1 99/100; dets 50/400; labels 3,629 CUDA / 3,628 OpenCL of 9,100; masks 1,240,141 CUDA / 1,239,727 OpenCL of 2,073,600 over tolerance (max ratio 47,350.3 / 47,350.4). PASS is on the ORT-neutral pin (TopK→2×Constant; ORT max_abs 0): dets 0/400, labels 0/9,100, top-1 100/100 on all three; masks 2/36/17 (CPU/OpenCL/CUDA) of 2,073,600, max ratio ≤ 2.6; IoU min 0.999800 at raw>=0; CUDA 0.984375 at raw>=0.5 (query 36, 63/64; CPU/OpenCL 1.000000). The pinned graph is a single-input counterfactual for this gate (TopK removed; 1,215 vs 1,216 nodes): not a TopK-correctness result, not a deployable substitute. Result-level equivalence, not bit-identity. |
| RT-DETRv4-s | detection | PASS | PASS | PASS | CUDA asymmetric-pad Conv remedy (explicit zero pre-pad in both CUDA Conv rows; symmetric path unchanged); full-model CUDA gate: boxes 0/1200 (max 7.93e-04), scores 0/300 (max 2.46e-05), labels 300/300 exact; 11 detections over 0.25, class ids [5,0,0,0,0,26,9,11,26,26,26]. |
| YOLOv8n detector | detection | PASS | PASS | PASS | Gate 2026-10-20; opset 18 pinned at export by the Neuriplo YOLO exporter (`export/detection/yolo/export.py --opset 18`); 301 nodes on disk / **260** dispatched (41 `Constant` folded); `images [1,3,640,640]` float32 → `output0 [1,84,8400]` float32; export sha256 `fc34eff309da7fbd723c597f9621bac0b8e55695cbd102f6807c18bda325edf5`; three legs exit 0, 260 dispatched on each: CPU run_ms 31,366.9 / wall 31.527 s, CUDA 751.398 ms / 1.347 s, OpenCL 362.951 ms / 0.806 s. `--tol 1e-3` on `output0` (705,600 elements): over_tol 0 on all three devices; max_abs / mean_abs / max_ratio CPU 0.000717163 / 7.75374e-07 / 0.00976039, CUDA 0.0010376 / 1.17403e-06 / 0.0178353, OpenCL 0.00134277 / 1.75999e-06 / 0.0277979. Host-side postprocessing, outside the ONNX gate: top-1 class agreement 8046/8400 (95.7857 %) on every device; the 354 disagreements are all argmax tie-breaks (max reference score at any such anchor 2.360e-05, zero above 0.01); at a fixed raw score > 0.25 all three devices give 47/47 anchors with identical sets and class-id diff 0. |
| YOLO11m-seg | detection + segmentation | PASS | PASS | PASS | Gate 2026-10-20; opset 18 pinned at export by the Ultralytics API from the local checkpoint (`yolo11m-seg.pt`); 515 nodes on disk / **467** dispatched (48 `Constant` folded); `images [1,3,640,640]` float32 → `output0 [1,116,8400]` and `output1 [1,32,160,160]` float32; export sha256 `9dc95d807fa6383a0f0360ca9c255ba4bf46b45db7f0522dccd8746d7f3dbe69`; three legs exit 0, 467 dispatched on each: CPU run_ms 499,296 / wall 500.37 s, CUDA 1,409.47 ms / 3.17 s, OpenCL 1,433.6 ms / 2.71 s. `--tol 1e-3`: over_tol 0 on all three devices and both outputs; max_abs / mean_abs / max_ratio — `output0` (974,400 elements) CPU 0.000671387 / 5.73567e-07 / 0.00858168, CUDA 0.00164795 / 1.30431e-06 / 0.025526, OpenCL 0.00160217 / 1.60192e-06 / 0.027492; `output1` (819,200 elements) CPU 5.72205e-06 / 3.36962e-07 / 0.0030517, CUDA 2.24113e-05 / 7.15764e-07 / 0.00893492, OpenCL 1.9908e-05 / 1.05817e-06 / 0.0120151. Host-side postprocessing, outside the ONNX gate: top-1 class agreement 8123/8400 (96.7024 %) on every device; fixed-threshold anchor sets at 0.25 and 0.5 are 50/50 with an identical (anchor,class) set (sym_diff 0). `output1` is the mask-prototype tensor, not per-instance masks; per-instance masks need NMS plus prototype assembly (host-side, outside the ONNX gate) and no per-query mask IoU was computed — the only `output1` statistics recorded are prototype-tensor ones (Pearson r 1.000000 over all 32 prototypes, 100 % sign agreement). |
| OWL-ViT | open-vocab detection | PASS | PASS | PASS | Gate 2026-10-20; opset 18 pinned at export by a worker-authored exporter (the Neuriplo repository has no OWL-ViT exporter); 1,418 nodes on disk / **1,071** dispatched (347 `Constant` folded); `pixel_values [1,3,768,768]` float32, `input_ids [3,16]` int64, `attention_mask [3,16]` int64 → `logits [1,576,3]` and `pred_boxes [1,576,4]` float32; export sha256 `935899b3c4884921c4582c59b2c96fe7bbcf3a2849347f8fb1c9715878c6eedf`. Provenance: the first provision produced a graph NERT's loader rejected (`Cast to=11` DOUBLE — NERT's `DataType` has no Float64 — and a `Reciprocal` outside the loader's accepted-op map); it was re-exported at two sites in the exporter only, the patched artifact proven bit-identical to the rejected one (ORT-vs-ORT and torch-vs-torch, five input sets, max abs delta 0, 0 differing elements) with the structural diff confined to the two patched constructs; the rejected artifact is preserved beside it (`bf9bd81c…`) and is not loadable by NERT. The gate's precondition re-confirmed at gate time: the patched artifact's ORT reference is byte-identical to the rejected artifact's reference for both outputs (max abs delta 0.0, 0 differing elements). Three legs exit 0, 1,071 dispatched on each: CPU run_ms 330,579 / wall 337.25 s, CUDA 736.355 ms / 7.82 s, OpenCL 461.558 ms / 7.45 s. `--tol 1e-3`: over_tol 0 on all three devices and both outputs; max_abs / mean_abs / max_ratio — `logits` (1,728 elements) CPU 0.000134468 / 1.36089e-05 / 0.0170852, CUDA 0.000162125 / 1.43129e-05 / 0.0219091, OpenCL 0.000442028 / 2.29426e-05 / 0.052925; `pred_boxes` (2,304 elements) CPU 1.48416e-05 / 2.43454e-07 / 0.00982289, CUDA 2.67625e-05 / 2.99385e-07 / 0.0177128, OpenCL 6.01709e-05 / 4.1826e-07 / 0.0458542. Host-side postprocessing, outside the ONNX gate: the per-query argmax over the 576 patch positions is the index set {303, 272, 144} on the reference and on every device; score deltas ≤ 2.25e-5, box deltas ≤ 5.96e-07, IoU ≥ 0.999997 at a gate-local IoU 0.9 — the Neuriplo repository has no production threshold for this model, so the 0.9 is the gate's own choice and is disclosed as such. The text inputs are a static [3,16] three-query fixture, not a general open-vocabulary interface (the OWLv2 sibling uses 16). |
| ResNet-50 | classification | PASS | PASS | PASS | Gate 2026-10-20; opset 18, converted from an opset-20 raw export by `onnx.version_converter` (the Neuriplo torchvision exporter has no opset argument); conversion does not prove semantic equivalence — the full-model gate is the equivalence evidence. 122 nodes on disk / **122** dispatched (zero `Constant` nodes); `input [1,3,224,224]` float32 → `output [1,1000]` float32; export sha256 `dc771684b16dd8123ffa1d68204e15a12c2127efdce629d581d675dbaaa185ce`; three legs exit 0, 122 dispatched on each: CPU run_ms 55,937.8 / wall 56.99 s, CUDA 540.998 ms / 2.13 s, OpenCL 152.329 ms / 1.57 s. `--tol 1e-3` on `output` (1,000 elements): over_tol 0 on all three devices; max_abs / mean_abs / max_ratio CPU 5.21541e-07 / 1.24643e-07 / 0.000459771, CUDA 7.7486e-07 / 1.85609e-07 / 0.000669164, OpenCL 1.78814e-06 / 3.95639e-07 / 0.00132994. Top-1 class agreement with the reference is 1/1 (100 %) on all three devices (class index 487) — n = 1, a weak statistic, not a robust agreement rate. Provenance: the Neuriplo exporter's `export_to_onnx` unconditionally replaces `model.fc` with a fresh unseeded `torch.nn.Linear(2048,1000)`, so the exported head carries no ImageNet training (exported `fc.weight` absmax 0.0220970809 = the 1/sqrt(2048) kaiming-uniform bound, against the checkpoint's 0.7371372) while the 53 `Conv` nodes do carry the ImageNet `IMAGENET1K_V1` weights with BatchNorm folded — the logits are not ImageNet-classifiable and the gate result is numerical equivalence, not semantic class correctness; the unseeded head makes the artifact not byte-reproducible, so only its recorded sha256 identifies the instance. The graph's pooling node is `ReduceMean`, which differs from the ResNet-18 `/tmp` export's `GlobalAveragePool`; the two must not be conflated. |
| ViTPose-base-simple | pose estimation | PASS | PASS | PASS | Gate 2026-10-20; opset **18 pinned at export** by the Neuriplo ViTPose exporter (`export/pose_estimation/vitpose/export_vitpose_to_onnx.py --model usyd-community/vitpose-base-simple --opset 18 --static-batch --image-height 256 --image-width 192`); 536 nodes on disk / **422** dispatched (114 `Constant` folded at load by the loader's read loop, `src/model_loader.cpp:1804-1812`, and never dispatched); `pixel_values [1,3,256,192]` float32 → `heatmaps [1,17,64,48]` float32 (52,224 elements); export sha256 `b72b99e52da700dbbef81c7dd5e677b40002338724dd7ec7430cb4ba84fee5c8` (HF `usyd-community/vitpose-base-simple`, resolved commit `a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45`; the stock exporter run first produced a non-self-contained artifact under torch 2.12's `dynamo=True` default, so the exporter was copied to `/tmp` and `dynamo=False` added at the call site only, recorded verbatim in the export provenance); three legs exit 0, 422 dispatched on each: CPU run_ms 106,612 / total_ms 106,616, CUDA 565.503 / 839.577, OpenCL 156.388 / 248.700 — these are the harness's own spans (`run_ms` = model run, `total_ms` = construct+run), not process elapsed time. `--tol 1e-3` scaled `tol*(1+abs(ref))` on `heatmaps`: over_tol 0 on all three devices; max_abs / mean_abs / max_ratio CPU 2.02656e-06 / 1.17238e-08 / 0.00144604, CUDA 2.02656e-06 / 9.30653e-09 / 0.00140438, OpenCL 3.30806e-06 / 1.91889e-08 / 0.00242397. The frozen fixture is bus.jpg → RGB → BILINEAR 256x192 → /255 → NCHW, sha256 `7da2b1afd99815b6250055e7c9261866b8b6b7b7b8d9f8ea60588ce7ba62040f`; the ORT 1.27.0 CPUExecutionProvider reference `ref_heatmaps.npy` sha256 `9bd99dec48bbe3156fffc92a713ecf0e279d868cbdd2000b45425acba67b0378` was independently re-derived bit-for-bit by the gate's read-only review (an ORT 1.27.0 CPUExecutionProvider re-run on that fixture, bit-identical array and sha256). Host-side postprocessing, OUTSIDE the ONNX gate: per-keypoint-channel argmax agreement 17/17 on every device; max abs Δscore at the reference's argmax 1.49011612e-06 CPU / 9.53674316e-07 CUDA / 1.60932541e-06 OpenCL. Per-device op gap: the only difference within the graph's op census is the folded `Constant`; the OpenCL table additionally lacks `Abs`, `Floor`, `GreaterOrEqual`, `Neg`, none of which occurs in this graph. No kernel row was added: the tables are unchanged at CPU 72 / CUDA 72 / OpenCL 68 / HIP 9, and no engine source file changed this boundary. |
| Depth-Anything-V2-Small | depth estimation | PASS | PASS | PASS | Gate 2026-10-20; opset **18 pinned at export** by the Neuriplo Depth-Anything-V2 exporter; 553 nodes on disk / **553** dispatched (zero `Constant` nodes); `pixel_values [1,3,518,518]` float32 → `predicted_depth [1,518,518]` float32 (268,324 elements); export sha256 `497acfffe67bffb5a00cc2aeaa11c91f033266385134ab12ebb26c36b9748492`; three legs exit 0, 553 dispatched on each: CPU run_ms 252205 / total_ms 252209, CUDA 1317.54 / 1548.05, OpenCL 503.735 / 554.784 — harness spans, not process wall time. `--tol 1e-3` scaled `tol*(1+\|ref\|)` on `predicted_depth`: over_tol 0 on all three devices; max_abs / mean_abs / max_ratio CPU 9.60827e-05 / 5.27289e-07 / 0.0381143, CUDA 9.63211e-05 / 7.20690e-07 / 0.0372235, OpenCL 1.08480e-04 / 1.72654e-06 / 0.0419957. Frozen fixture sha256 `6a50426dda0294930cce1398dbeaad33a8cc3563a6d04faf3102464a95d378c4`; ORT 1.27.0 CPUExecutionProvider reference `ref_predicted_depth.npy` sha256 `f305d87277c0286583004743ceda0a39db7e7303e9baa51a12bc973b3c25c2ce`. Host-side postprocessing, OUTSIDE the ONNX gate: nearest/farthest-pixel agreement with the reference on CPU/CUDA, Spearman rho 1.000000000, mean abs deviation < 1e-6. OpenCL path: the gate exposed a `Resize` presence-rejection defect for no-op `antialias=0`/`exclude_outside=0`, fixed architect-authorized in `src/opencl/resize.cpp` only (no ABI/table change). |
| VideoMAE | video classification | PASS | PASS | PASS | Gate 2026-10-20; opset **18 pinned at export** (legacy TorchScript exporter; HF `MCG-NJU/videomae-base-finetuned-kinetics` @ `488eb9a0565f257b32866000305c8178965eb9f6`); 521 nodes on disk / **407** dispatched (114 `Constant` folded); `pixel_values [1,16,3,224,224]` float32 → `logits [1,400]` float32 (400 elements); export sha256 `14fe0aced2ee6e547f4d8866829f2d9fd474d53da5fb9c6414ee1470c05cb586`; fixture `videomae.npy` sha256 `537c5e905549a0afe1329f63d6e76e26cd2b5ec3e574ce602fea68245ac2e0fc`; ORT 1.27.0 CPUExecutionProvider reference `ref_logits_videomae.npy` sha256 `20efdf6ca3e3586cd5ac4362cd1071d2b256cf2ac2736ab64a368bd77f1c95e2` (argmax 143). `--tol 1e-3` scaled `tol*(1+\|ref\|)`: over_tol **0/400** on all three devices; max_abs / mean_abs / max_ratio CPU 1.3113021850585938e-06 / 3.292784094810486e-07 / 0.0009440241920040157, CUDA 1.2516975402832031e-06 / 3.4138327464461327e-07 / 0.0008659470031916535, OpenCL 1.9669532775878906e-06 / 4.022219218313694e-07 / 0.0009925095604235309. All legs exit 0. |
| ViViT | video classification | PASS | PASS | PASS | Gate 2026-10-20; opset **18 pinned at export** (legacy TorchScript exporter; HF `google/vivit-b-16x2-kinetics400` @ `8a7171a57f79b9aaa58bc8d977c002a0ea0f0d42`); final export 1,152 nodes on disk / **797** dispatched (355 `Constant` folded), `If`=0 `Identity`=3; the raw export (sha256 `6300d2776dd8e1ccd5997d90750f87a974e082971f420dc60e99241c550b38d1`, 1,152 nodes, `If`=2 `Identity`=1) is preserved, and raw vs folded logits are bit-identical on the frozen fixture (`max_abs_diff` 0.0); `pixel_values [1,32,3,224,224]` float32 → `logits [1,400]` float32 (400 elements); final export sha256 `24d5690ab5423bad63ddc782bfd7a30cbfcb80d3a43cf3fd1c59f83685d42d39`; fixture `vivit.npy` sha256 `78157408ea97c35a0c262dff2acb78ed614164ef1ac844a80b4442dc3a467536`; ORT 1.27.0 CPUExecutionProvider reference `ref_logits_vivit.npy` sha256 `9fd575da47044ef880271f61bd02f06e8da201f7c222226f11c307dec0903f4b` (argmax 382). `--tol 1e-3` scaled: over_tol **0/400** on all three devices; max_abs / mean_abs / max_ratio CPU 7.3909759521484375e-06 / 1.6508117550984025e-06 / 0.004278399711686194, CUDA 8.344650268554688e-06 / 1.782802864909172e-06 / 0.004494117344208187, OpenCL 3.8623809814453125e-05 / 9.126970544457436e-06 / 0.031353039818606473. All legs exit 0. |
| TimeSformer | video classification | PASS | PASS | PASS | Gate 2026-10-20; opset **18 pinned at export** (legacy TorchScript exporter; HF `facebook/timesformer-base-finetuned-k400` @ `8aaf40ea7d3d282dcb0a5dea01a198320d15d6c0`); 1,289 nodes on disk / **904** dispatched (385 `Constant` folded); `pixel_values [1,8,3,224,224]` float32 → `logits [1,400]` float32 (400 elements); export sha256 `f0256c3770adba4df11f4a3d5e20596830b88784bdc7686137d8dc495f40335a`; fixture `timesformer.npy` sha256 `2d10ec9b9ffd0050511900efe11f05ebca326b37e8c8459e69635bccba4eb564`; ORT 1.27.0 CPUExecutionProvider reference `ref_logits_timesformer.npy` sha256 `150a1393cc307890ee026e228e4c680a652af46e33e4edccd889eb0e3961e078` (argmax 304). `--tol 1e-3` scaled: over_tol **0/400** on all three devices; max_abs / mean_abs / max_ratio CPU 3.337860107421875e-06 / 8.425139822065831e-07 / 0.0018960872446111409, CUDA 3.814697265625e-06 / 1.0447530075907708e-06 / 0.002557859409865797, OpenCL 6.67572021484375e-06 / 1.395069994032383e-06 / 0.003530061276282632. All legs exit 0. |
| 6 missing-export entries | all families | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | EdgeCrafter detection; EdgeCrafter segmentation; EdgeCrafter pose; RF-DETR detection other sizes; RF-DETR segmentation other sizes; RF-DETR keypoint other sizes. See inventory; export-first. |
| Image understanding, Gaussian Splatting | — | OUT-OF-SCOPE | OUT-OF-SCOPE | OUT-OF-SCOPE | No ONNX contract. |

Video fixture/reference provenance: one frozen canonical `(1,32,3,224,224)` buffer from seed `20261020` (`numpy.random.default_rng`), with F=16/32/8 prefixes for VideoMAE/ViViT/TimeSformer; references are ORT 1.27.0 CPUExecutionProvider. Harness-provenance caveat: the prebuilt `nert_gate_{cpu,cuda,opencl}` binaries predate HEAD `47bfe8f` (`bench: typed fixtures for R6/Tier-5 rows`, a bench-only commit); TimeSformer's dispatched op set contains no `Resize`, so its numerics are unaffected, and CPU/CUDA/OpenCL agree with each other and the reference (0/400 over tolerance). No engine source file changed.

## Release scoreboard

| Status | Count | Entries |
| --- | ---: | --- |
| PASS | 23 | Five primary models + eight converted variants + RT-DETRv4-s + RF-DETR seg-nano-576 (SN deviation) + YOLOv8n detector + YOLO11m-seg + OWL-ViT + ResNet-50 + ViTPose-base-simple + Depth-Anything-V2-Small + entry 31 (optical-flow small) + entry 29 (video: VideoMAE/ViViT/TimeSformer) |
| Open | 1 | Grounding DINO (CPU/CUDA task-equivalence PASS; OpenCL backend gap) |
| Missing export | 6 | EdgeCrafter detection; EdgeCrafter segmentation; EdgeCrafter pose; RF-DETR detection other sizes; RF-DETR segmentation other sizes; RF-DETR keypoint other sizes |
| Out of scope | 2 | Image understanding, Gaussian Splatting |
| **Total** | **32** | 30 in scope |

23 + 1 + 6 = 30 in-scope entries. RAFT-small (entry 31) is the
twenty-second PASS and entry 29 (video) is the twenty-third: the entry closes
once, with VideoMAE, ViViT and TimeSformer recorded as its three gated models.
The release gate is zero open entries and
zero missing exports among those 30; HIP is outside the gate. The catalog is
not complete and 0.2.0 is not release-ready.

The PASS count assumes one gated representative per catalog entry closes that
entry (YOLOv8n detector for entry 17, YOLO11m-seg for entry 18, OWL-ViT for
entry 25, ResNet-50 for entry 30, RAFT-small for entry 31); other sizes of
those groups are not gated and do not move the scoreboard. Entry 29 is counted
once, as a single entry, even though all three of its gated models are recorded
in the evidence table. That is the ledger's prepared-decomposition
working assumption and an open maintainer decision, so the counts depend on it.

## Open items

Closed and dropped from this list: the Grounding DINO CPU/CUDA
task-equivalence gate, the RF-DETR seg-nano-576 SN-DEV ruling and the
RT-DETRv4-s full-model CUDA gate.

1. **Grounding DINO OpenCL cell** — a recorded backend gap (the handle path
   demands 11,028.5 MiB on a 5,803.3 MiB device). Whether the release gate
   admits a documented GAP-with-deviation is still an open maintainer
   decision. Boundary 7 (video) did not touch this cell; it remains Open, not
   PASS.
2. **Missing exports** — provision, probe, gate and review the 6 remaining
   entries: EdgeCrafter detection, EdgeCrafter segmentation, EdgeCrafter pose,
   RF-DETR detection other sizes, RF-DETR segmentation other sizes and
   RF-DETR keypoint other sizes.
   VideoMAE/ViViT/TimeSformer closed 2026-10-20 (three PASS rows above).

## Non-kernel limits

- **Opset policy.** NERT loads declared opsets 11 and above. Opsets 11–18
  canonicalize to the opset-18 IR; above 18 every node is checked against a
  generated effective-schema compatibility table and rejected if it uses an
  unimplemented attribute, value, or behavior. Models below 11 are still
  pinned or converted, and conversion does not prove semantic equivalence.
  See `specs/2026-10-21-opset-range/` and
  `specs/2026-10-21-opset-newer-than-18/`.
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
