<subagent sessionID="ses_eac8f539dffe6K53Ynl8lRXJ0b" state="completed">
# NERT release-gate catalog inventory

Date: 2026-10-19. Read-only inspection of `/home/oli/repos/neuriplo-tasks`, `/home/oli/repos/neuriplo`, and `/home/oli/repos/nert`. No files were modified and no builds were run.

**Counting convention.** Table A is one row per `TaskFactory` descriptor (18 rows), with every accepted literal/wildcard alias expanded in-cell. Table B is the release scoreboard: **32 catalog entries**, one per representative model/variant group. Aliases that resolve to the same graph are collapsed in B; size-only variants that have no artifact are grouped. NERT-only FoundationPose/RF-DETR artifacts are listed separately and are **not counted** in C.

**Path aliases used below (all absolute):**

- `BR = /home/oli/repos/neuriplo/build-parity`
- `OP = /tmp/opencode/opset18`
- `TMP = /tmp/opencode`
- `NT = /home/oli/repos/neuriplo-tasks`
- `NF = /home/oli/repos/nert`

## Critical corrections / ambiguities found before the matrix

1. **`coverage.md:142` is stale for RT-DETR.** It says the DETR family has no local export. A local RT-DETRv4-small export exists at `/home/oli/repos/neuriplo-tasks/weights/rtv4_hgnetv2_s_model.onnx` (opset **16**, 996 nodes, inputs `images` FLOAT `[N,3,640,640]` + `orig_target_sizes` INT64 `[?,2]`). NERT rejects it because the loader requires exactly opset 18 (`NF/src/model_loader.cpp:110`, `:1760-1784`). It is a conversion candidate, not a no-export family.
2. **The ResNet-18 PASS row is fixture-dependent.** `coverage.md:52,154` and `opencl-kernel-matrix/validation.md:32` describe the 49-node `/tmp/opencode/resnet18_export/resnet18.onnx`, which uses `GlobalAveragePool`. The Neuriplo parity fixture `BR/native_parity_fixture/resnet18.onnx` is a **different** 49-node export using `ReduceMean` + `Reshape`. `ReduceMean` is absent from `NF/src/opencl/table.cpp:61-99`, so that fixture would fail OpenCL with `no kernel for op 'ReduceMean'`. The release gate must pin which ResNet-18 file it means.
3. **Grounding DINO cannot be upgraded by OpenCL work.** It loads and is CPU/CUDA op-type complete, but both gates fail at `/model/model/Clip` on int64 token ids; the Clip row is float32-only (`coverage.md:159`, `2026-10-18-opencl-kernel-matrix/validation.md:37`).
4. **The loadable RF-DETR seg-small export is not the `opset18/` file.** `TMP/rfdetr-seg-small-18.onnx` (1,785 nodes, 1,255 non-Constant) loads and was gated. `OP/rfdetr_seg_small-18.onnx` (1,880 nodes) is the deep-infer duplicate blocked on dynamic `Slice` (`coverage.md:141`; the line says 1,879 nodes, inspected file has 1,880).
5. **`/home/oli/repos/neuriplo-tasks/model.onnx` is a 0-byte invalid artifact.** It is not a model and should not be counted.
6. **Factory accept-set ≠ runtime dispatch-set for size-suffixed names.** The factory accepts `yolo26n`, `yolov10n`, `yolonas_s`, etc. via `startsWith("yolo")`, but `ObjectDetectionTask::detectModelType` (`NT/src/object_detection/object_detection_task.cpp:179-219`) only special-cases the exact family token (`"yolo26"`, `"yolov10"`, `"yolonas"`). Passing `yolo26n` therefore routes to `YOLO_STANDARD`, not `YOLO_NMS_FREE`. For the gate, pass the family token and put the size in the model path. The same caution applies to `raft-large` (only `raft` is accepted), `videomae-base...` (only `videomae`), and `rfdetr-nano` (only `rfdetr`).
7. **RT-DETRv4 is not a factory alias.** `DetrDetection` accepts `rtdetr`, `rtdetrv2`, `rtdetrul`, `rtdetrultralytics`, `rfdetr`, `dfine*`, `deim*` (`NT/src/core/task_factory.cpp:129-137`) but **not** `rtdetrv4`, even though the exporter supports `--version v4` (`NT/export/detection/rtdetr/export.py:537`). The v4 graph must be declared as `rtdetr` (its outputs `labels`/`boxes`/`scores` match `RT_DETR_STYLE`).

### ModelType enums found

| Family | Enum values | Header |
| --- | --- | --- |
| Detection | `UKNOWN`, `YOLO_STANDARD`, `YOLO_V4`, `YOLO_NMS_FREE`, `YOLO_NAS`, `YOLO_V7_E2E`, `RF_DETR`, `RT_DETR_UL`, `RT_DETR_STYLE`, `EDGECRAFTER` | `NT/include/neuriplo/tasks/object_detection/object_detection_task.hpp:32-44` |
| Instance segmentation | `YOLO_SEG`, `YOLO_V10_SEG`, `YOLO_26_SEG`, `RF_DETR_SEG`, `EDGECRAFTER_SEG` | `.../instance_segmentation/instance_segmentation_task.hpp:16-22` |
| Pose | `RFDETRPOSE`, `UNKNOWN`, `YOLO`, `VITPOSE`, `EDGECRAFTER` | `.../pose_estimation/pose_estimation_task.hpp:28` |
| Depth | `DEPTH_ANYTHING_V2`, `YOLO_DEPTH` | `.../depth_estimation/depth_estimation_task.hpp:18` |
| Open-vocab | `Unknown`, `OWLV2`, `OWLVIT`, `GroundingDino` | `.../open_vocab_detection/open_vocab_detection_task.hpp:19` |
| Classification | `TORCHVISION`, `TENSORFLOW`, `VIT` | `.../classification/classification_task.hpp:32-36` |
| Optical flow | `RAFT` | `.../optical_flow/optical_flow_task.hpp:15-17` |
| Video | `VIDEOMAE`, `VIVIT`, `TIMESFORMER` | `.../video_classification/video_classification_task.hpp:26` |
| Gaussian splatting | `LGM`, `GRM` | `.../gaussian_splatting/gaussian_splatting_task.hpp:29-32` |
| Image understanding | none; `model_name` is ignored | `NT/src/image_understanding/image_understanding_task.cpp:7-11` |

---

## 0. Local ONNX inventory

The inventory below records every catalog-relevant ONNX file inspected. Node counts are ONNX graph nodes; the non-Constant count is given where it differs. “Loadable” means the file was converted/loaded and shape-inferred per `coverage.md:104-113`.

### 0a. Catalog-relevant exports

| Path | Size (B) | Opset | Nodes | Inputs (dtype/shape) | Outputs (dtype/shape) |
| --- | ---: | ---: | ---: | --- | --- |
| `BR/native_parity_fixture/resnet18.onnx` (+ `.data`) | 89,298 | 18 | 49 | `input` FLOAT `[1,3,224,224]` | `output` FLOAT `[1,1000]` |
| `TMP/resnet18_export/resnet18.onnx` | 46,748,551 | 18 | 49 | `input` FLOAT `[1,3,224,224]` | `output` FLOAT `[1,1000]` |
| `BR/native_yolo_parity_fixture/yolo26n.onnx` | 9,765,332 | 18 | 485 | `images` FLOAT `[1,3,640,640]` | `output0` FLOAT `[1,300,6]` |
| `TMP/owlv2-base-patch16.onnx` | 618,672,505 | 18 | 1,219 | `input_ids` INT64 `[2,16]`, `pixel_values` FLOAT `[1,3,960,960]`, `attention_mask` INT64 `[2,16]` | `logits` FLOAT `[1,3600,2]`, `pred_boxes` FLOAT `[1,3600]`, plus 8 extra float outputs |
| `TMP/rfdetr-seg-small-18.onnx` | 123,537,326 | 18 | 1,785 | `input` FLOAT `[1,3,384,384]` | `dets` FLOAT `[1,100,4]`, `labels` FLOAT `[1,100,91]`, `masks` FLOAT `[1,100,96,96]` |
| `TMP/seg-export/model/rfdetr-seg-small.onnx` | 123,378,609 | 17 | 1,784 | same | same |
| `TMP/raft/raft_large.onnx` | 21,348,820 | 18 | 1,836 | `image1`/`image2` FLOAT `[1,3,368,496]` | `flow` FLOAT `[1,2,368,496]` |
| `OP/groundingdino-18.onnx` | 969,952,242 | 18 | 31,828 (19,224 non-Const) | `pixel_values` FLOAT `[N,3,800,1333]`, `input_ids`/`attention_mask`/`token_type_ids` INT64 `[N,256]` | `logits` FLOAT `[N,?,?]`, `pred_boxes` FLOAT `[N,?,4]` |
| `OP/rfdetr_keypoint_preview-18.onnx` | 156,719,972 | 18 | 3,543 (2,355 non-Const) | `input` FLOAT `[1,3,576,576]` | `dets` FLOAT `[1,100,4]`, `labels` FLOAT `[1,100,2]`, `keypoints` FLOAT `[1,100,34,8]` |
| `OP/rfdetr_keypoint_3542-18.onnx` | 156,719,972 | 18 | 3,543 | same | same (duplicate of preview) |
| `OP/rfdetr_seg_nano_576-18.onnx` | 132,150,021 | 18 | 1,711 (1,216 non-Const) | `input` FLOAT `[1,3,576,576]` | `dets` FLOAT `[1,100,4]`, `labels` FLOAT `[1,100,91]`, `masks` FLOAT `[1,100,144,144]` |
| `OP/rfdetr_nano_1101-18.onnx` | 115,260,108 | 18 | 1,435 (1,002 non-Const) | `input` FLOAT `[1,3,640,640]` | `dets` FLOAT `[1,300,4]`, `labels` FLOAT `[1,300,91]` |
| `OP/rfdetr_seg_small-18.onnx` (deep-infer duplicate) | 130,047,839 | 18 | 1,880 (1,309 non-Const) | `input` FLOAT `[1,3,384,384]` | `dets` FLOAT `[1,100,4]`, `labels` FLOAT `[1,100,91]`, `masks` FLOAT `[1,100,96,96]` |
| `OP/scorer-18.onnx` | 63,911,834 | 18 | 210 (145 non-Const) | `A`/`B` FLOAT `[B,6,160,160]` | `score` FLOAT `[B]` |
| `OP/ref2-18.onnx` | 68,193,414 | 18 | 365 (250 non-Const) | `A`/`B` FLOAT `[B,6,160,160]` | `trans` FLOAT `[B,3]`, `rot` FLOAT `[B,3]` |
| `OP/refiner-18.onnx` | 68,197,078 | 18 | 384 (276 non-Const) | same | same |
| `OP/rw-18.onnx` | 68,196,363 | 18 | 379 (272 non-Const) | same | same |
| `OP/sw-18.onnx` | 63,928,048 | 18 | 301 (214 non-Const) | same | same |
| `OP/yolo26s-18.onnx` | 38,366,622 | 18 | 391 | `images` FLOAT `[1,3,640,640]` | `output0` FLOAT `[1,300,6]` |
| `OP/yolo26x-18.onnx` | 223,245,590 | 18 | 606 | `images` FLOAT `[1,3,640,640]` | `output0` FLOAT `[1,300,6]` |
| `OP/yolo26n_depth-18.onnx` | 20,753,236 | 18 | 348 (328 non-Const) | `images` FLOAT `[1,3,768,768]` | `output0` FLOAT `[1,1,768,768]` |
| `OP/yolov8n_pose-18.onnx` | 13,497,914 | 18 | 267 | `images` FLOAT `[1,3,640,640]` | `output0` FLOAT `[1,56,8400]` |
| `OP/yolo11s-18.onnx` | 38,030,880 | 18 | 331 (321 non-Const) | `images` FLOAT `[1,3,640,640]` | `output0` FLOAT `[1,84,8400]` |
| `OP/yolo26s_pose-18.onnx` | 41,818,330 | 18 | 412 (401 non-Const) | `images` FLOAT `[1,3,640,640]` | `output0` FLOAT `[1,56,8400]` |
| `NT/weights/rtv4_hgnetv2_s_model.onnx` | 41,760,774 | **16** | 996 | `images` FLOAT `[N,3,640,640]`, `orig_target_sizes` INT64 `[?,2]` | `labels` INT64 `[?,300]`, `boxes` FLOAT `[?,300,4]`, `scores` FLOAT `[?,300]` |
| `NT/weights/rtv4_hgnetv2_s_model.pth` | 169,278,400 | — | — | checkpoint, not ONNX | — |
| `NT/model.onnx` | **0** | — | 0 | invalid / empty | invalid / empty |

Opset-17 originals still present: `TMP/ref2.onnx` 68,169,420 B/363 nodes; `TMP/refiner.onnx` 68,172,171/382; `TMP/rw.onnx` 68,171,649/377; `TMP/sw.onnx` 63,913,433/300; `TMP/scorer.onnx` 63,900,998/209. All are opset 17 and superseded by the `OP/*-18.onnx` conversions.

Other duplicates: `TMP/owlv2-data/owlv2-base-patch16.onnx` is byte-identical in size/nodes/IO to `TMP/owlv2-base-patch16.onnx`; `TMP/rfdetr-data/rfdetr-seg-small-18.onnx` is identical to `TMP/rfdetr-seg-small-18.onnx`.

`TMP/ref.onnx` (214,525 B, opset 18, 156 nodes) is the historical FoundationPose-style `A`/`B` → `trans`/`rot` graph misattributed to RAFT; it is **not** a catalog model (`coverage.md:58`).

### 0b. Debug/probe ONNX (not catalog models)

All `sub_*` probes share input `input` FLOAT `[1,3,384,384]`; `cast6.probe.onnx` is `x` FLOAT `[4]` → `y` INT32 `[4]`. All are opset 18.

| File | Size (B) | Nodes | Output (dtype/shape) |
| --- | ---: | ---: | --- |
| `TMP/cast6.probe.onnx` | 72 | 1 | `y` INT32 `[4]` |
| `TMP/sub_add1.onnx` | 94,397,630 | 895 | `/transformer/Add_1_output_0` FLOAT `[?,1024,2]` |
| `TMP/sub_add2.onnx` | 94,502,573 | 948 | `/transformer/Add_2_output_0` FLOAT `[?,100,2]` |
| `TMP/sub_cat5.onnx` | 94,399,075 | 903 | `/transformer/Concat_5_output_0` FLOAT `[?,1024,4]` |
| `TMP/sub_cat6.onnx` | 94,495,598 | 918 | `/transformer/Concat_6_output_0` FLOAT `[?,100,?]` |
| `TMP/sub_cat7.onnx` | 94,504,000 | 956 | `/transformer/Concat_7_output_0` FLOAT `[?,100,4]` |
| `TMP/sub_cat8.onnx` | 94,505,107 | 962 | `/transformer/Concat_8_output_0` FLOAT `[?,100,4]` |
| `TMP/sub_dec_in.onnx` | 104,002 | 8 | `/transformer/Expand_3_output_0` FLOAT `[?,100,256]` |
| `TMP/sub_dec_norm.onnx` | 119,149,636 | 1,663 | `/transformer/decoder/norm/LayerNormalization_output_0` FLOAT `[?,100,256]` |
| `TMP/sub_dec_ref.onnx` | 95,326,431 | 1,105 | `/transformer/decoder/ref_point_head/layers.1/Add_output_0` FLOAT `[?,?,256]` |
| `TMP/sub_enc_final.onnx` | 87,757,726 | 705 | `.../layernorm_3/LayerNormalization_output_0` FLOAT `[4,257,384]` |
| `TMP/sub_enc_l0.onnx` | 4,647,581 | 84 | `.../layer.0/norm2/LayerNormalization_output_0` FLOAT `[4,257,384]` |
| `TMP/sub_enc_l11.onnx` | 83,020,653 | 690 | `.../layer.11/norm2/LayerNormalization_output_0` FLOAT `[4,257,384]` |
| `TMP/sub_enc_l5.onnx` | 40,271,010 | 358 | `.../layer.5/norm2/LayerNormalization_output_0` FLOAT `[4,257,384]` |
| `TMP/sub_exp1.onnx` | 5,139 | 19 | `/transformer/Exp_1_output_0` FLOAT `[?,100,2]` |
| `TMP/sub_ge.onnx` | 94,495,420 | 917 | `/transformer/GatherElements_output_0` FLOAT `[?,100,?]` |
| `TMP/sub_layer0.onnx` | 101,368,095 | 1,281 | `.../layers.0/norm3/LayerNormalization_output_0` FLOAT `[?,100,256]` |
| `TMP/sub_layer1.onnx` | 107,294,441 | 1,408 | `.../layers.1/norm3/LayerNormalization_output_0` FLOAT `[?,100,256]` |
| `TMP/sub_layer2.onnx` | 113,220,787 | 1,535 | `.../layers.2/norm3/LayerNormalization_output_0` FLOAT `[?,100,256]` |
| `TMP/sub_layer3.onnx` | 119,147,133 | 1,662 | `.../layers.3/norm3/LayerNormalization_output_0` FLOAT `[?,100,256]` |
| `TMP/sub_mul1.onnx` | 94,396,712 | 890 | `/transformer/Mul_1_output_0` FLOAT `[?,1024,2]` |
| `TMP/sub_mul4.onnx` | 94,501,497 | 942 | `/transformer/Mul_4_output_0` FLOAT `[?,100,2]` |
| `TMP/sub_mul5.onnx` | 94,501,644 | 943 | `/transformer/Mul_5_output_0` FLOAT `[?,100,2]` |
| `TMP/sub_rph_add0.onnx` | 95,062,298 | 1,102 | `/transformer/decoder/ref_point_head/layers.0/Add_output_0` FLOAT `[?,?,256]` |
| `TMP/sub_rph_mm0.onnx` | 95,060,891 | 1,101 | `/transformer/decoder/ref_point_head/layers.0/MatMul_output_0` FLOAT `[?,?,256]` |
| `TMP/sub_rph_relu.onnx` | 95,062,539 | 1,103 | `/transformer/decoder/ref_point_head/Relu_output_0` FLOAT `[?,?,256]` |
| `TMP/sub_slice6.onnx` | 4,098 | 13 | `/transformer/Slice_6_output_0` FLOAT `[?,0,4]` |
| `TMP/sub_slice8.onnx` | 94,496,499 | 923 | `/transformer/Slice_8_output_0` FLOAT `[?,100,?]` |
| `TMP/sub_slice9.onnx` | 94,496,499 | 923 | `/transformer/Slice_9_output_0` FLOAT `[?,100,?]` |
| `TMP/sub_topk.onnx` | 93,955,903 | 872 | `TopK_output_0` FLOAT `[?,100]`, `TopK_output_1` INT64 `[?,100]` |

---

## A. Catalog families — every `TaskFactory` descriptor and alias

| # | Factory descriptor (file:line) | Family | Accepted model-type tokens (expanded) | Representative model(s) | ONNX runtime contract? | Exporter (file:line) | Local export (opset) | NERT load |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `YoloSegmentation` `task_factory.cpp:82-87` | Instance seg | `yoloseg`, `yolov10seg`, `yolo26seg`, `rfdetrseg`, `yolo*seg` (`yolov5seg`, `yolov8seg`, `yolo11seg`, `yolo12seg`, `yolo26nseg`, …) | RF-DETR seg-small; RF-DETR seg-nano-576; YOLO26-seg variants | Yes — dets + mask coeffs/protos, or NMS-free dets+protos | `NT/export/segmentation/rfdetr/export.py:26-37`; YOLO seg is manual (`NT/export/segmentation/InstanceSegmentation.md:1-12`) | `TMP/rfdetr-seg-small-18.onnx` (18); `OP/rfdetr_seg_nano_576-18.onnx` (18); YOLO seg **MISSING** | Yes for RF-DETR seg; YOLO seg no export |
| 2 | `EdgeCrafterSegmentation` `task_factory.cpp:88-89` | Instance seg | `ecseg*` (`ecseg_s`, `ecseg_m`, `ecseg_l`, `ecseg_x`) | EdgeCrafter ecseg_s..x | Yes — `images`+`orig_target_sizes` → `labels`/`boxes`/`scores`/`masks` (`NT/export/segmentation/edgecrafter/README.md:31-40`) | `NT/export/segmentation/edgecrafter/README.md:1-30` (upstream script, no local `.py`) | **MISSING** | No |
| 3 | `RfDetrPose` `task_factory.cpp:92-97` | Pose | `rfdetr*pose`, `rfdetr*keypoint`, `rfdetr*kpt` (`rfdetrpose`, `rfdetr-keypoint`, `rfdetr-kpt`, `rfdetrkeypointpreview`, …) | RF-DETR keypoint-preview | Yes — `dets`/`labels`/`keypoints` | `NT/export/pose_estimation/rfdetr/export_keypoint.py:19-20` | `OP/rfdetr_keypoint_preview-18.onnx` (18); duplicate 3542 | Yes |
| 4 | `YoloPose` `task_factory.cpp:98-100` | Pose | `yolo*pose` (`yolopose`, `yolov5pose`, `yolov8pose`, `yolo11pose`, `yolo26pose`, `yolo26s-pose`, `yolov8n-pose`) | YOLOv8n-pose, YOLO26s-pose | Yes — `output0` `[N,56,8400]` (or `[N,8400,56]` for v5) (`NT/export/pose_estimation/PoseEstimation.md:5-12`) | `NT/export/pose_estimation/yolo_pose/export.py:221-247` (default opset 12) | `OP/yolov8n_pose-18.onnx`, `OP/yolo26s_pose-18.onnx` (18); other sizes **MISSING** | Yes |
| 5 | `EdgeCrafterPose` `task_factory.cpp:101-102` | Pose | `ecpose*` (`ecpose_s`, `ecpose_m`) | EdgeCrafter ecpose_s/m | Yes — `labels`/`scores`/`keypoints` (`NT/export/pose_estimation/edgecrafter/README.md:31-40`) | `NT/export/pose_estimation/edgecrafter/README.md:1-30` | **MISSING** | No |
| 6 | `VitPose` `task_factory.cpp:103-104` | Pose | `vitpose` (exact only) | ViTPose-base-simple | Yes — heatmaps `[N,17,64,48]` | `NT/export/pose_estimation/vitpose/export_vitpose_to_onnx.py:137-149` (default opset 14) | **MISSING** | No |
| 7 | `YoloDepthEstimation` `task_factory.cpp:107-110` | Depth | `yolo*depth` (`yolodepth`, `yolo26depth`, `yolo26ndepth`, `yolo26n-depth`, …) | YOLO26n-depth | Yes — depth `[B,1,H,W]` (`NT/export/depth_estimation/yolo_depth/README.md:20-25`) | Ultralytics CLI (`NT/export/depth_estimation/yolo_depth/README.md:5-18`) | `OP/yolo26n_depth-18.onnx` (18); other variants **MISSING** | Yes |
| 8 | `YoloDetection` `task_factory.cpp:113-114` | Detection | `yolo*` generic (`yolo`, `yolov4`, `yolov5`, `yolov6`, `yolov7`, `yolov7e2e`, `yolov8`, `yolov9`, `yolov10`, `yolo11`, `yolo12`, `yolo26`, `yolonas`, size-suffixed variants) | YOLO26n/s/x, YOLO11s | Yes — `output0` `[1,300,6]` (NMS-free) or `[1,84,8400]` (standard) (`NT/export/detection/ObjectDetection.md:36-53`) | `NT/export/detection/yolo/export.py:662-710` (default opset 12) | `BR/native_yolo_parity_fixture/yolo26n.onnx` (18); `OP/yolo26s-18.onnx`, `OP/yolo26x-18.onnx`, `OP/yolo11s-18.onnx` (18); other variants **MISSING** | Yes for the four local graphs |
| 9 | `EdgeCrafterDetection` `task_factory.cpp:115-116` | Detection | `ecdet*` (`ecdet_s`, `ecdet_m`, `ecdet_l`, `ecdet_x`) | EdgeCrafter ecdet_s..x | Yes — `images`+`orig_target_sizes` → `labels`/`boxes`/`scores` (`NT/export/detection/edgecrafter/README.md:31-37`) | `NT/export/detection/edgecrafter/README.md:1-30` | **MISSING** | No |
| 10 | `EdgeCrafterFamily` `task_factory.cpp:117-128` | Detection / seg / pose | `edgecrafter*`; dispatch by substring: contains `seg` → seg, `pose` → pose, else detection | EdgeCrafter base | Yes — family-specific contracts above | Same EdgeCrafter READMEs | **MISSING** | No |
| 11 | `DetrDetection` `task_factory.cpp:129-137` | Detection | `rtdetr`, `rtdetrv2`, `rtdetrul`, `rtdetrultralytics`, `rfdetr`, `dfine*`, `deim*` | RT-DETRv4-s (declared `rtdetr`); RF-DETR nano-1101; D-FINE/DEIM | Yes — `labels`/`boxes`/`scores`; RF-DETR `dets`/`labels` | `NT/export/detection/rtdetr/export.py:494-544`; `NT/export/detection/rfdetr/export.py:9-20` (default opset 17) | `NT/weights/rtv4_hgnetv2_s_model.onnx` (**16**, not loadable); `OP/rfdetr_nano_1101-18.onnx` (18); other sizes **MISSING** | No for v4 until converted; yes for RF-DETR nano |
| 12 | `OpenVocabDetection` `task_factory.cpp:140-147` | Open-vocab | `owlv2`, `owlvit`, `groundingdino` | OWLv2-base-patch16; Grounding DINO-base; OWL-ViT | Yes — OWLv2 `logits`/`pred_boxes`; Grounding DINO `logits`/`pred_boxes` | `NT/export/open_vocab_detection/owlv2/export_owlv2_to_onnx.py:136-146` (default 17); `.../groundingdino/export_groundingdino_to_onnx.py:221-228` (default 17) | `TMP/owlv2-base-patch16.onnx` (18); `OP/groundingdino-18.onnx` (18); OWL-ViT **MISSING** | Yes OWLv2; Grounding DINO loads but gate blocked; OWL-ViT no export |
| 13 | `Classification` `task_factory.cpp:150-159` | Classification | `torchvisionclassifier`, `tensorflowclassifier`, `vitclassifier`, `resnet*` (`resnet18`, `resnet50`, …), `*tensorflow*` | ResNet-18; ResNet-50; ViT-base; TensorFlow ResNet-50 | Yes — logits `[N,C]` | `NT/export/classification/torchvision/export_torchvision_classifier.py:38-46` (no opset pin); `.../vit/onnx/export.py:34` (hardcoded 17); `.../tensorflow/export_tf_saved_model_classifier.py:1-13` | `BR/native_parity_fixture/resnet18.onnx` (18); `TMP/resnet18_export/resnet18.onnx` (18); others **MISSING** | Yes for ResNet-18 (with fixture caveat); others no export |
| 14 | `DepthEstimation` `task_factory.cpp:162-165` | Depth | `*depthanythingv2*` (`depthanythingv2`, `depthanythingv2small`, `depth-anything-v2-small`) | Depth-Anything-V2-Small/Base | Yes — `pixel_values` → `predicted_depth` `[B,H,W]` (`NT/export/depth_estimation/DepthEstimation.md:11-18`) | `NT/export/depth_estimation/depth_anything_v2/export_depth_anything_v2_to_onnx.py:138-152` (default 14) | **MISSING** | No |
| 15 | `GaussianSplatting` `task_factory.cpp:168-174` | Gaussian splatting | `lgm`, `grm`, `gaussiansplatting`, `lgmmini`, `*splat*` | LGM/LGM-mini; GRM | Yes conditional — exporter maps image `[N,3,H,W]` → gaussians `[N,G,14]`; rendering is not ONNX (`NT/export/gaussian_splatting/GaussianSplatting.md:1-30`) | `NT/export/gaussian_splatting/lgm/export_lgm_to_onnx.py:124-134` (default 17; LGM only) | **MISSING** | No |
| 16 | `VideoClassification` `task_factory.cpp:177-185` | Video classification | `videomae`, `vivit`, `timesformer` (exact only; HF IDs are not accepted) | VideoMAE-base; ViViT-b-16x2; TimeSformer-base | Yes — 5D input `[N,T,3,H,W]` → logits `[N,C]` (`NT/export/video_classification/VideoClassification.md:5-12`) | `.../videomae/export_videomae_to_onnx.py:136-148` (14); `.../vivit/export_vivit_to_onnx.py:141-153` (14); `.../timesformer/export_timesformer_to_onnx.py:146-158` (14) | **MISSING** | No |
| 17 | `OpticalFlow` `task_factory.cpp:188-190` | Optical flow | `raft` (exact only; `raft-large` is not accepted) | RAFT-large; RAFT-small | Yes — `image1`/`image2` → `flow` `[B,2,H,W]` | `NT/export/optical_flow/raft/raft_exporter.py:198-226` (hardcodes opset 16 at `:174`) | `TMP/raft/raft_large.onnx` (18); RAFT-small **MISSING** | Yes RAFT-large; RAFT-small no export |
| 18 | `ImageUnderstanding` `task_factory.cpp:193-201` | Image understanding | `gemma4`, `gemma`, `llama`, `llamacpp`, `imageunderstanding` (exact only) | Gemma 4 E4B IT; Llama-family GGUF | **No** — GGUF + llama.cpp; no ONNX graph (`NT/export/image_understanding/ImageUnderstanding.md:1-11`) | No ONNX exporter; GGUF download only (`ImageUnderstanding.md:13-40`) | N/A — GGUF local (see D) | N/A — llama.cpp backend, not NERT |

---

## B. Status matrix — 32 catalog entries

Legend: **PASS** = full-model gate recorded; **OPS** = graph loads and CPU/CUDA op types are covered, but no full-model gate; **GAP(n/m)** = `n` missing op types / `m` post-fold node touches; **BLOCKED** = graph exists but cannot complete a gate; **MISSING-EXPORT** = no loadable export; **OUT-OF-SCOPE** = no ONNX contract or not a Neuriplo task. HIP is a NERT-internal column: Neuriplo selects only CPU or CUDA (`neuriplo/backends/nert/src/NertInfer.cpp:81-87`) and `coverage.md:209-215` records no OpenCL/HIP device selection.

| # | Representative model | Family / canonical type | CPU | CUDA | OpenCL | HIP | Evidence status | Blocker / note | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ResNet-18 (gated `/tmp` export) | classification / `resnet18` | PASS | PASS | PASS | GAP(5/24) | **PASS** | HIP lacks `Conv`(20), `GlobalAveragePool`, `MaxPool`, `Flatten`, `Gemm`; Neuriplo parity fixture is a different export with `ReduceMean` (see F1) | `coverage.md:52,154`; `validation.md:29`; `opencl-kernel-matrix/validation.md:32` |
| 2 | YOLO26n | detection / `yolo26` | PASS | PASS | GAP(6/21) | GAP(21/197) | **PASS CPU/CUDA** | OpenCL: `Split=12`, `GatherElements=3`, `Resize=2`, `TopK=2`, `Mod=1`, `ReduceMax=1` | `coverage.md:53,155,65-66`; `n1g-yolo-gpu/validation.md` R-6 |
| 3 | OWLv2-base-patch16 | open-vocab / `owlv2` | PASS | PASS | PASS | GAP(24/453) | **PASS** | HIP missing 24 types/453 nodes; no OpenCL gap | `coverage.md:54,156`; `validation.md:32,34`; `opencl-kernel-matrix/validation.md:36` |
| 4 | RF-DETR seg-small | instance seg / `rfdetrseg` | PASS | PASS | GAP(14/84) | GAP(33/701) | **PASS CPU/CUDA** | 14 OpenCL types; 33 HIP types | `coverage.md:55,157`; `2026-10-18-model-gates/validation.md:32` |
| 5 | RAFT-large | optical flow / `raft` | PASS | PASS | GAP(7/116) | GAP(19/501) | **PASS CPU/CUDA** | `GridSample=48`, `Split=48`, `InstanceNormalization=15`, `Range=2`, `Pad=1`, `ReduceSum=1`, `Tile=1` | `coverage.md:56,158`; `2026-10-18-model-gates/validation.md:36` |
| 6 | Grounding DINO-base | open-vocab / `groundingdino` | BLOCKED | BLOCKED | GAP(26/1,960) | GAP(45/15,336) | **BLOCKED** | `/model/model/Clip` int64 token ids vs float32-only Clip; ORT ref exists but gate cannot pass | `coverage.md:159`; `opencl-kernel-matrix/validation.md:37` |
| 7 | RF-DETR keypoint-preview | pose / `rfdetr-keypoint` | OPS | OPS | GAP(15/133) | GAP(33/1,486) | **OPS** | No reference gate; 15 OpenCL types | `coverage.md:118` |
| 8 | RF-DETR seg-nano-576 | instance seg / `rfdetrseg` | OPS | OPS | GAP(14/83) | GAP(33/675) | **OPS** | No reference gate; 14 OpenCL types | `coverage.md:119` |
| 9 | RF-DETR nano-1101 | detection / `rfdetr` | OPS | OPS | GAP(13/71) | GAP(31/567) | **OPS** | No reference gate; 13 OpenCL types | `coverage.md:120` |
| 10 | YOLO26s | detection / `yolo26` | OPS | OPS | GAP(7/14) | GAP(17/176) | **OPS** | No reference gate; 7 OpenCL types | `coverage.md:122` |
| 11 | YOLO26x | detection / `yolo26` | OPS | OPS | GAP(7/23) | GAP(17/247) | **OPS** | No reference gate; 7 OpenCL types | `coverage.md:123` |
| 12 | YOLO26n-depth | depth / `yolo26n-depth` | OPS | OPS | GAP(6/21) | GAP(11/139) | **OPS** | `Clip`, `ConvTranspose`, `Exp`, `Pow`, `Resize`, `Split` | `coverage.md:124` |
| 13 | YOLOv8n-pose | pose / `yolov8n-pose` | OPS | OPS | GAP(2/10) | GAP(7/110) | **OPS** | `Resize=2`, `Split=8`; 2-row unlock | `coverage.md:125` |
| 14 | YOLO11s | detection / `yolo11` | OPS | OPS | GAP(2/12) | GAP(8/132) | **OPS** | `Resize=2`, `Split=10`; 2-row unlock | `coverage.md:126` |
| 15 | YOLO26s-pose | pose / `yolo26s-pose` | OPS | OPS | GAP(2/13) | GAP(8/164) | **OPS** | `Resize=2`, `Split=11`; 2-row unlock | `coverage.md:127` |
| 16 | RT-DETRv4-s | detection / `rtdetr` (v4 not an alias) | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | Local file is opset **16**; NERT loader requires 18. Needs conversion, then gate. `coverage.md:142` is stale | Inspected `NT/weights/rtv4_hgnetv2_s_model.onnx`; `NF/src/model_loader.cpp:110` |
| 17 | YOLO detector variants v5/v6/v7/v8/v9/v10/v12 | detection / `yolo*` | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | No local ONNX; some checkpoints local (see D) | `coverage.md:142,161` |
| 18 | YOLO segmentation variants | instance seg / `yolo*seg` | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | No local ONNX; `yolo11m-seg.pt` local only | `coverage.md:142,161`; `NT/export/segmentation/InstanceSegmentation.md:1-12` |
| 19 | EdgeCrafter detection (ecdet_s..x) | detection / `ecdet*` | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | No local ONNX; upstream export only | `coverage.md:142,161`; `NT/export/detection/edgecrafter/README.md` |
| 20 | EdgeCrafter segmentation (ecseg_s..x) | instance seg / `ecseg*` | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | No local ONNX; requirements.md marks unexported seg future work | `coverage.md:142,161`; `2026-10-17-task-models/requirements.md:35-36` |
| 21 | EdgeCrafter pose (ecpose_s/m) | pose / `ecpose*` | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | No local ONNX; requirements.md marks unexported pose future work | `coverage.md:142,161`; `requirements.md:35-36` |
| 22 | RF-DETR detection small/medium/large/xlarge/2xlarge | detection / `rfdetr` | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | No local ONNX; rfdetr import currently broken (F10) | `coverage.md:142,161`; inspected `rfdetr` import |
| 23 | RF-DETR segmentation medium/large/xlarge/2xlarge | instance seg / `rfdetrseg` | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | No local ONNX | `coverage.md:142,161` |
| 24 | RF-DETR keypoint nano/small/medium/large/xlarge | pose / `rfdetr-keypoint` | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | No local ONNX; exporter falls back to preview (`export_keypoint.py:44-50`) | `coverage.md:142,161`; exporter source |
| 25 | OWL-ViT | open-vocab / `owlvit` | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | No exporter script in `NT/export/open_vocab_detection/` | `task_factory.cpp:142`; directory listing |
| 26 | ViTPose-base-simple | pose / `vitpose` | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | Exporter exists; no local ONNX/checkpoint | `coverage.md:142,161`; `export_vitpose_to_onnx.py` |
| 27 | Depth-Anything-V2-Small/Base | depth / `depthanythingv2*` | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | Exporter exists; no local ONNX/checkpoint | `coverage.md:142,161`; `export_depth_anything_v2_to_onnx.py` |
| 28 | LGM/LGM-mini/GRM | gaussian splatting / `lgm`, `grm` | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | No local ONNX; `diffusers` missing; GRM has no dedicated exporter | `coverage.md:142,161`; inspected environment |
| 29 | VideoMAE/ViViT/TimeSformer | video classification | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | No local ONNX/checkpoints | `coverage.md:142,161`; exporters exist |
| 30 | Classification ViT/TensorFlow/other torchvision | classification | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | Only ResNet-18 has local ONNX | `coverage.md:142,161`; exporters exist |
| 31 | RAFT-small | optical flow / `raft` | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | MISSING-EXPORT | **MISSING-EXPORT** | No local ONNX; exporter hardcodes opset 16 | `coverage.md:56,142`; `raft_exporter.py:174` |
| 32 | Image understanding (Gemma 4/Llama) | image understanding | OUT-OF-SCOPE | OUT-OF-SCOPE | OUT-OF-SCOPE | OUT-OF-SCOPE | **OUT-OF-SCOPE** | GGUF/llama.cpp; no ONNX runtime contract | `NT/export/image_understanding/ImageUnderstanding.md:1-11`; `neuriplo/docs/backends.yaml:227-241` |

### B-extra: NERT-only artifacts, not Neuriplo catalog entries

| Artifact | CPU | CUDA | OpenCL | HIP | Status | Reason | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FoundationPose scorer | OPS | OPS | GAP(2/2) | GAP(10/98) | **OPS / OUT-OF-SCOPE** | No `TaskFactory` 6-D pose family; NERT-internal | `coverage.md:121` |
| FoundationPose `ref2`, `refiner`, `rw`, `sw` | BLOCKED | BLOCKED | BLOCKED | BLOCKED | **BLOCKED / OUT-OF-SCOPE** | Dynamic `Slice` parameters in attention reshape; shape inference fails | `coverage.md:140`; `2026-10-18-opset18-conversions/validation.md:23` |
| Deep-infer RF-DETR seg-small duplicate | BLOCKED | BLOCKED | BLOCKED | BLOCKED | **BLOCKED** | Dynamic `Slice` parameters; the NERT-side export loads | `coverage.md:141`; `opset18-conversions/validation.md:23` |
| RF-DETR keypoint 3542 duplicate | OPS | OPS | GAP(15/133) | GAP(33/1,486) | **OPS** | Same folded graph as preview (2,355 nodes) | `coverage.md:118,129-130` |

---

## C. Counts

Counted over the 32 catalog entries in Table B. NERT-only rows are excluded; they are listed separately in B-extra.

| Category | Count | Entries |
| --- | ---: | --- |
| Total catalog entries | **32** | All rows 1–32 |
| Covered on all three CPU/CUDA/OpenCL | **2** | ResNet-18 (gated `/tmp` export), OWLv2-base-patch16 |
| CPU+CUDA PASS but OpenCL gap | **3** | YOLO26n, RF-DETR seg-small, RAFT-large |
| OPS-only (loads, CPU/CUDA op-complete, no full gate) | **9** | RF-DETR keypoint-preview, RF-DETR seg-nano-576, RF-DETR nano-1101, YOLO26s, YOLO26x, YOLO26n-depth, YOLOv8n-pose, YOLO11s, YOLO26s-pose |
| Blocked on loader/op | **1** | Grounding DINO (int64 Clip) |
| Blocked on export/opset | **16** | RT-DETRv4-s (opset 16); YOLO detector variants; YOLO segmentation variants; EdgeCrafter det; EdgeCrafter seg; EdgeCrafter pose; RF-DETR det other sizes; RF-DETR seg other sizes; RF-DETR kp other sizes; OWL-ViT; ViTPose; Depth-Anything-V2; Gaussian LGM/GRM; video 3; classification non-ResNet; RAFT-small |
| Out-of-scope | **1** | Image understanding (GGUF/llama.cpp) |

**ResNet-18 caveat:** the “all three” count is for the gated `/tmp/opencode/resnet18_export/resnet18.onnx` (`GlobalAveragePool`). If the release gate pins the Neuriplo parity fixture `BR/native_parity_fixture/resnet18.onnx`, ResNet-18 becomes **CPU+CUDA PASS, OpenCL GAP(1/1)** (`ReduceMean`), changing the counts to 1 all-three and 4 CPU+CUDA-with-OpenCL-gap.

**NERT-only counts (not in the 32):** OPS = 2 (FoundationPose scorer, RF-DETR keypoint 3542 duplicate); BLOCKED = 2 groups (FoundationPose ref2/refiner/rw/sw, deep-infer RF-DETR duplicate).

---

## D. Provisioning feasibility for every missing export

“Offline” means no network needed on this host; “download” means a checkpoint/HF model must be fetched; “impossible here” means no existing exporter or no viable local path. Commands are given but were **not run**.

| Missing export | Exporter (file:line) | Dependency / checkpoint needed | Network? | Local alternative | Verdict | Exact command (do not run) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO detector variants v5/v6/v7/v8/v9/v10/v12 | `NT/export/detection/yolo/export.py:662-710` (default opset 12) | `ultralytics` (installed 8.4.113) for v8–v12/26; v5/v6/v7 need repo clones; checkpoints | Offline for `yolo26s.pt`, `yolo26x.pt`, `yolo11s.pt`, `yolo26n.pt`; download for others | `/home/oli/repos/model-optimizations/yolo26s.pt`, `yolo26x.pt`, `yolo11s.pt`; `/home/oli/repos/neuriplo/yolo26n.pt` | **Provisionable offline** for those four; **with download** for v5/v6/v7/v8/v9/v10/v12 | `python3 NT/export/detection/yolo/export.py --model /home/oli/repos/model-optimizations/yolo26s.pt --version 26 --format onnx --opset 18 --output-dir /tmp/opencode/yolo26s` |
| YOLO segmentation variants | No dedicated script; `NT/export/segmentation/InstanceSegmentation.md:1-12` uses Ultralytics CLI | `ultralytics`; `-seg.pt` checkpoints | Offline for `yolo11m-seg.pt`; download for other sizes | `/home/oli/repos/tritonic/yolo11m-seg.pt` (45.4 MB) | **Provisionable offline** for yolo11m-seg; **with download** for others | `yolo export model=/home/oli/repos/tritonic/yolo11m-seg.pt format=onnx opset=18 imgsz=640` |
| EdgeCrafter detection `ecdet_s..x` | `NT/export/detection/edgecrafter/README.md:1-30` (upstream `tools/deployment/export_onnx.py`) | Clone EdgeCrafter; `ecdetseg/requirements.txt`; `onnx onnxsim onnxscript`; `.pth` checkpoint | **Yes** (repo + checkpoint download) | None found | **Provisionable with download** | `git clone --depth 1 https://github.com/Intellindust-AI-Lab/EdgeCrafter && cd EdgeCrafter && curl -L -o ecdet_s.pth https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_s.pth && python tools/deployment/export_onnx.py -c configs/ecdet/ecdet_s.yml -r ecdet_s.pth --check --simplify` |
| EdgeCrafter segmentation `ecseg_s..x` | `NT/export/segmentation/edgecrafter/README.md:1-30` | Same + `ecseg_s.pth` | **Yes** | None found | **Provisionable with download** | `... export_onnx.py -c configs/ecseg/ecseg_s.yml -r ecseg_s.pth --check --simplify` |
| EdgeCrafter pose `ecpose_s/m` | `NT/export/pose_estimation/edgecrafter/README.md:1-30` | Same + `ecpose_s.pth` | **Yes** | None found | **Provisionable with download** | `... export_onnx.py -c configs/ecpose/ecpose_s_coco.yml -r ecpose_s.pth --check --simplify` |
| RF-DETR detection other sizes | `NT/export/detection/rfdetr/export.py:9-20` (default 17) | `rfdetr` package; currently import fails with transformers 4.35 (`BackboneConfigMixin` missing); weights auto-download | **Yes** (dependency + weights) | Existing ONNX: nano-1101; seg-small; seg-nano; keypoint-preview | **Provisionable with download + dependency fix**; **impossible here without upgrading `transformers`** | `python3 NT/export/detection/rfdetr/export.py --model_type small --opset_version 18 --simplify --output_dir /tmp/opencode/rfdetr_small` |
| RF-DETR segmentation other sizes | `NT/export/segmentation/rfdetr/export.py:26-37` | Same | **Yes** | seg-small, seg-nano-576 | **Provisionable with download + dependency fix** | `python3 NT/export/segmentation/rfdetr/export.py --model_type medium --opset_version 18 --simplify --output_dir /tmp/opencode/rfdetr_seg_medium` |
| RF-DETR keypoint other sizes | `NT/export/pose_estimation/rfdetr/export_keypoint.py:19-20` | Same; exporter may silently fall back to `RFDETRKeypointPreview` (`:44-50`) | **Yes** | keypoint-preview | **Provisionable with download + dependency fix**; not a true new size unless the class exists | `python3 NT/export/pose_estimation/rfdetr/export_keypoint.py --model_type small --opset_version 18 --output_dir /tmp/opencode/rfdetr_kp_small` |
| OWL-ViT | **No exporter script** in `NT/export/open_vocab_detection/` | Would need a new HF `google/owlvit-*` exporter; no local checkpoint | **Yes** + new code | None | **Impossible here** (no exporter in repo) | No existing command; would require a new script (out of scope for this release) |
| ViTPose-base-simple | `NT/export/pose_estimation/vitpose/export_vitpose_to_onnx.py:137-149` (default 14) | `transformers` with `VitPoseForPoseEstimation`; HF `usyd-community/vitpose-base-simple` | **Yes** (model download; transformers 4.35 may be too old) | None found | **Provisionable with download** (+ transformers upgrade) | `python3 NT/export/pose_estimation/vitpose/export_vitpose_to_onnx.py --model usyd-community/vitpose-base-simple --opset 18 --output /tmp/opencode/vitpose.onnx --test` |
| Depth-Anything-V2-Small/Base | `NT/export/depth_estimation/depth_anything_v2/export_depth_anything_v2_to_onnx.py:138-152` (default 14) | HF `depth-anything/Depth-Anything-V2-Small-hf`; transformers | **Yes** | Local HF cache has Depth-Anything-**V3** (`DA3-SMALL`), not V2 | **Provisionable with download** | `python3 NT/export/depth_estimation/depth_anything_v2/export_depth_anything_v2_to_onnx.py --model depth-anything/Depth-Anything-V2-Small-hf --opset 18 --output /tmp/opencode/depth_anything_v2_small.onnx --test` |
| LGM / LGM-mini | `NT/export/gaussian_splatting/lgm/export_lgm_to_onnx.py:124-134` (default 17) | `diffusers` (not installed), `transformers`, `trust_remote_code`; HF `dylanebert/LGM` | **Yes** (pip install + model download) | None found | **Provisionable with download**, but exporter uses a single 3-channel image while the task requires 4 views (F12) | `pip install diffusers && python3 NT/export/gaussian_splatting/lgm/export_lgm_to_onnx.py --model dylanebert/LGM --opset 18 --output /tmp/opencode/lgm.onnx --test` |
| GRM | No dedicated exporter | HF `GaussianWorld/GRM`; the LGM script assumes `pipeline.unet` | **Yes** + exporter work | None | **Impossible here** without a GRM exporter | No existing command |
| VideoMAE | `NT/export/video_classification/videomae/export_videomae_to_onnx.py:136-148` (default 14) | HF `MCG-NJU/videomae-base-finetuned-kinetics` | **Yes** | None | **Provisionable with download** | `python3 NT/export/video_classification/videomae/export_videomae_to_onnx.py --opset 18 --output /tmp/opencode/videomae.onnx --test` |
| ViViT | `.../vivit/export_vivit_to_onnx.py:141-153` (default 14) | HF `google/vivit-b-16x2-kinetics400` | **Yes** | None | **Provisionable with download** | `python3 NT/export/video_classification/vivit/export_vivit_to_onnx.py --opset 18 --output /tmp/opencode/vivit.onnx --test` |
| TimeSformer | `.../timesformer/export_timesformer_to_onnx.py:146-158` (default 14) | HF `facebook/timesformer-base-finetuned-k400` | **Yes** | None | **Provisionable with download** | `python3 NT/export/video_classification/timesformer/export_timesformer_to_onnx.py --opset 18 --output /tmp/opencode/timesformer.onnx --test` |
| Classification non-ResNet | Torchvision `NT/export/classification/torchvision/export_torchvision_classifier.py:38-46` (no opset pin); ViT `.../vit/onnx/export.py:34` (hardcoded 17); TF `.../tensorflow/export_tf_saved_model_classifier.py:1-13` | torchvision/timm weights or TensorFlow; ViT HF `google/vit-base-patch16-224` | **Yes** | ResNet-18 weights cached (`~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`) | **Provisionable with download** for new classifiers; ResNet-18 already local | `python3 NT/export/classification/torchvision/export_torchvision_classifier.py --library torchvision --model resnet50 --export_format onnx --output_onnx /tmp/opencode/resnet50.onnx` (then pin/convert to opset 18) |
| RAFT-small | `NT/export/optical_flow/raft/raft_exporter.py:198-226`; opset hardcoded 16 at `:174` | torchvision RAFT-small weights (download); no `--opset` flag | **Yes** for weights | RAFT-large weights cached (`~/.cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth`); small not cached | **Provisionable with download**; must convert 16→18 or patch the script | `python3 NT/export/optical_flow/raft/raft_exporter.py --model-type small --format onnx --output-dir /tmp/opencode/raft_small` then `python3 -c "import onnx; from onnx import version_converter; m=onnx.load('/tmp/opencode/raft_small/raft_small.onnx'); m=version_converter.convert_version(m,18); onnx.save(m,'/tmp/opencode/raft_small/raft_small-18.onnx')"` |
| RT-DETRv4-s | `NT/export/detection/rtdetr/export.py:494-544`; local ONNX already exists | Local checkpoint `NT/weights/rtv4_hgnetv2_s_model.pth`; only conversion needed | **No** | `NT/weights/rtv4_hgnetv2_s_model.onnx` (opset 16) | **Provisionable offline** (16→18) | `python3 -c "import onnx; from onnx import version_converter; m=onnx.load('/home/oli/repos/neuriplo-tasks/weights/rtv4_hgnetv2_s_model.onnx'); m=version_converter.convert_version(m,18); onnx.save(m,'/tmp/opencode/opset18/rtv4_hgnetv2_s-18.onnx')"` |
| Image understanding Gemma 4 / Llama | No ONNX exporter; `NT/export/image_understanding/ImageUnderstanding.md:1-11` | llama.cpp backend + GGUF + mmproj | **No** | `/home/oli/repos/ai-code-assistant-playground/models/gemma-4-E4B-it-Q4_K_M.gguf` (5.34 GB) and `mmproj-gemma4-E4B-F16.gguf` (990 MB) | **Provisionable offline** for Gemma 4 E4B; **not an ONNX export** | `neuriplo-infer --model_type gemma4 --model_path "/home/oli/repos/ai-code-assistant-playground/models/gemma-4-E4B-it-Q4_K_M.gguf|mmproj=/home/oli/repos/ai-code-assistant-playground/models/mmproj-gemma4-E4B-F16.gguf" --image input.jpg --prompt "Describe this image."` |
| FoundationPose components | No exporter in `NT/export/`; NERT artifacts only | NERT-side 6-D pose pipeline | No | `OP/scorer-18.onnx`, `OP/ref2-18.onnx`, etc. | **Impossible here / OUT-OF-SCOPE** | No catalog command |

**Local checkpoint notes.** `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` and `raft_large_C_T_SKHT_V2-ff5fadd5.pth` exist. HF cache contains full `owlv2-base-patch16` and `grounding-dino-base` weights; it does **not** contain OWL-ViT, ViTPose, Depth-Anything-V2, VideoMAE, ViViT, TimeSformer, LGM, or GRM. Local YOLO checkpoints: `/home/oli/repos/neuriplo/yolo26n.pt`, `/home/oli/repos/model-optimizations/{yolo26n,yolo26s,yolo26x,yolo11s,yolo11x}.pt`, `/home/oli/repos/tritonic/yolo11m-seg.pt`, `/home/oli/repos/neuriplo-infer/models/trt-video/yolo26s-pose.pt`. No local `yolo26n-depth.pt`.

---

## E. Prioritized work list — shared OpenCL rows, grouped by unlock set

### E1. Work groups

“Fully unlocks for gating” means all missing op types in that group are implemented and a full-model gate is then run; it does **not** mean PASS by itself. Shared rows should be implemented once and reused across groups.

| Group | Models in group | Union of missing OpenCL op types | What the group unlocks |
| --- | --- | --- | --- |
| **G0 — complete** | OWLv2; ResNet-18 (gated `/tmp` export) | none | Already OpenCL PASS |
| **G1 — YOLO core** (highest leverage) | YOLO26n, YOLO26s, YOLO26x, YOLO11s, YOLOv8n-pose, YOLO26s-pose | `Split`, `Resize`, `GatherElements`, `TopK`, `ReduceMax`, `Mod`, `Tile` | Split+Resize alone unlocks **YOLO11s**, **YOLOv8n-pose**, **YOLO26s-pose** for gating. Adding GatherElements+TopK+ReduceMax+Mod unlocks **YOLO26n**. Adding Tile unlocks **YOLO26s/x**. |
| **G2 — YOLO depth extras** | YOLO26n-depth | `Clip`, `ConvTranspose`, `Exp`, `Pow` (+ G1 `Split`, `Resize`) | Unlocks **YOLO26n-depth** |
| **G3 — DETR/RF-DETR core** (P0 set) | RF-DETR seg-small, RF-DETR seg-nano-576, RF-DETR nano-1101, RF-DETR keypoint-preview | `Sqrt`, `ReduceSum`, `Cos`, `GridSample`, `Sin`, `Exp`, `Not`, `Less`, `Identity`, `ScatterND` (+ G1 `Split`, `Resize`, `GatherElements`, `TopK`, `ReduceMax`, `Tile`) | Unlocks **RF-DETR seg-small**, **seg-nano-576**, **nano-1101**, **keypoint-preview** for gating; keypoint additionally needs `Identity`+`ScatterND`. |
| **G4 — RAFT extras** (P0 set) | RAFT-large | `InstanceNormalization`, `Range`, `Pad` (+ G3 `GridSample`, `ReduceSum`; G1 `Split`, `Tile`) | Unlocks **RAFT-large** |
| **G5 — Grounding DINO** | Grounding DINO | `Clip`, `CumSum`, `EyeLike`, `Log`, `Max`, `Min`, `Or`, `Squeeze` + G1/G3 shared set | Completes OpenCL op coverage, but the gate remains **BLOCKED** on int64 `Clip`; no PASS until the dtype issue is fixed |
| **G6 — RT-DETRv4** | RT-DETRv4-s | `Clip`, `GatherElements`, `GridSample`, `Log`, `Pad`, `Pow`, `ReduceMax`, `ReduceMean`, `ReduceSum`, `Resize`, `Split`, `Sqrt`, `Squeeze`, `Tile`, `TopK` | Unlocks **RT-DETRv4-s** for gating after the 16→18 conversion |
| **G7 — ResNet fixture** | Neuriplo `BR/native_parity_fixture/resnet18.onnx` | `ReduceMean` (1 node) | Only needed if the release gate pins the Neuriplo fixture instead of the `/tmp` export (F1) |
| **G8 — no export** | YOLO v5/v6/v7/v8/v9/v10/v12, YOLO seg, EdgeCrafter det/seg/pose, RF-DETR other sizes, OWL-ViT, ViTPose, Depth-Anything-V2, Gaussian, video, classification non-ResNet, RAFT-small | no OpenCL work possible | Provisioning first (D) |

### E2. Known P0 sets — exact coverage

| P0 model | Missing OpenCL types (node touches) | Minimal incremental unlock |
| --- | --- | --- |
| **YOLO26n** | `Split=12`, `GatherElements=3`, `Resize=2`, `TopK=2`, `Mod=1`, `ReduceMax=1` — 6 types/21 nodes | All six rows needed; no subset unlocks YOLO26n |
| **RF-DETR seg-small** | `Sqrt=48`, `ReduceSum=5`, `Split=5`, `Cos=4`, `GridSample=4`, `Sin=4`, `Exp=3`, `Not=3`, `Tile=3`, `GatherElements=1`, `Less=1`, `ReduceMax=1`, `Resize=1`, `TopK=1` — 14 types/84 nodes | All 14 rows needed |
| **RAFT-large** | `GridSample=48`, `Split=48`, `InstanceNormalization=15`, `Range=2`, `Pad=1`, `ReduceSum=1`, `Tile=1` — 7 types/116 nodes | All seven rows needed |
| **2-row quick win** | `Split`, `Resize` | Unlocks **YOLO11s** (2/12), **YOLOv8n-pose** (2/10), **YOLO26s-pose** (2/13) for gating |
| **YOLO26s/x** | YOLO26n set + `Tile=2` | 7 types/14 and 7/23 |
| **YOLO26n-depth** | `Split=11`, `Resize=5`, `Clip=1`, `ConvTranspose=1`, `Exp=2`, `Pow=1` | 6 types/21 |

### E3. Every catalog variant that loads today — exact OpenCL gap

Computed from the local graphs with `Constant` excluded, matching the coverage counts. These are the converted-table rows plus the primary models.

| Model | OpenCL missing types (node touches) | HIP missing (touches) |
| --- | --- | --- |
| ResNet-18 (`/tmp` gated export) | none | 5/24: `Conv=20`, `GlobalAveragePool=1`, `MaxPool=1`, `Flatten=1`, `Gemm=1` |
| ResNet-18 (`BR` fixture) | `ReduceMean=1` | 5/24 (same as above, with `ReduceMean` instead of `GlobalAveragePool`) |
| YOLO26n | 6/21: `Split=12`, `GatherElements=3`, `Resize=2`, `TopK=2`, `Mod=1`, `ReduceMax=1` | 21/197 |
| OWLv2 | none | 24/453 |
| RF-DETR seg-small | 14/84: `Sqrt=48`, `ReduceSum=5`, `Split=5`, `Cos=4`, `GridSample=4`, `Sin=4`, `Exp=3`, `Not=3`, `Tile=3`, `GatherElements=1`, `Less=1`, `ReduceMax=1`, `Resize=1`, `TopK=1` | 33/701 |
| RAFT-large | 7/116: `GridSample=48`, `Split=48`, `InstanceNormalization=15`, `Range=2`, `Pad=1`, `ReduceSum=1`, `Tile=1` | 19/501 |
| Grounding DINO | 26/1,960: `Clip=13`, `Cos=33`, `CumSum=8`, `EyeLike=1`, `GatherElements=1`, `GridSample=48`, `Identity=44`, `InstanceNormalization=4`, `Less=12`, `Log=12`, `Max=255`, `Min=255`, `Mod=19`, `Not=34`, `Or=1`, `Pad=26`, `Range=470`, `ReduceMax=13`, `ReduceSum=30`, `Resize=4`, `ScatterND=115`, `Sin=33`, `Split=9`, `Squeeze=513`, `Tile=6`, `TopK=1` | 45/15,336 |
| RF-DETR keypoint-preview | 15/133: `Cos=4`, `Exp=3`, `GatherElements=2`, `GridSample=8`, `Identity=3`, `Less=1`, `Not=3`, `ReduceMax=1`, `ReduceSum=10`, `ScatterND=17`, `Sin=4`, `Split=10`, `Sqrt=60`, `Tile=6`, `TopK=1` | 33/1,486 |
| RF-DETR seg-nano-576 | 14/83: `Cos=4`, `Exp=3`, `GatherElements=2`, `GridSample=4`, `Less=1`, `Not=3`, `ReduceMax=1`, `ReduceSum=5`, `Resize=1`, `Sin=4`, `Split=5`, `Sqrt=48`, `Tile=1`, `TopK=1` | 33/675 |
| RF-DETR nano-1101 | 13/71: `Cos=4`, `Exp=3`, `GatherElements=2`, `GridSample=2`, `Less=1`, `Not=3`, `ReduceMax=1`, `ReduceSum=3`, `Sin=4`, `Split=3`, `Sqrt=42`, `Tile=2`, `TopK=1` | 31/567 |
| YOLO26s | 7/14: `GatherElements=2`, `Mod=1`, `ReduceMax=1`, `Resize=2`, `Split=4`, `Tile=2`, `TopK=2` | 17/176 |
| YOLO26x | 7/23: `GatherElements=2`, `Mod=1`, `ReduceMax=1`, `Resize=2`, `Split=13`, `Tile=2`, `TopK=2` | 17/247 |
| YOLO26n-depth | 6/21: `Clip=1`, `ConvTranspose=1`, `Exp=2`, `Pow=1`, `Resize=5`, `Split=11` | 11/139 |
| YOLOv8n-pose | 2/10: `Resize=2`, `Split=8` | 7/110 |
| YOLO11s | 2/12: `Resize=2`, `Split=10` | 8/132 |
| YOLO26s-pose | 2/13: `Resize=2`, `Split=11` | 8/164 |
| FoundationPose scorer (out-of-scope) | 2/2: `ReduceMean=1`, `Squeeze=1` | 10/98 |
| RT-DETRv4-s (opset16, conversion first) | 15/112: `Clip=9`, `GatherElements=3`, `GridSample=9`, `Log=1`, `Pad=1`, `Pow=12`, `ReduceMax=1`, `ReduceMean=25`, `ReduceSum=3`, `Resize=2`, `Split=11`, `Sqrt=12`, `Squeeze=16`, `Tile=4`, `TopK=3` | 28/467 |
| Deep-infer RF-DETR duplicate (blocked) | 15/119: adds `Squeeze=24`, `Split=17` vs seg-small | 34/748 |
| FoundationPose ref2/refiner/rw/sw (blocked, out-of-scope) | 4–5 types/13–25 each: `Identity`, `Mod`, `ReduceMean`, `Sqrt`, `Squeeze` | 14–16 types/151–196 each |

### E4. Shared-row leverage

| Op | Loadable models missing it (count) | Total node touches | Notes |
| --- | ---: | ---: | --- |
| `Split` | 14 | 160 | Highest shared row; no single model is fully unlocked by it alone |
| `Resize` | 11 | 25 | With `Split`, unlocks YOLO11s + both pose variants |
| `Tile` | 9 | 27 | YOLO26s/x, all RF-DETR, RAFT, Grounding DINO, RT-DETR |
| `ReduceMax` | 9 | 21 | YOLO26n/s/x, all RF-DETR, Grounding DINO, RT-DETR |
| `GatherElements` | 9 | 18 | YOLO26n/s/x, all RF-DETR, Grounding DINO, RT-DETR |
| `TopK` | 9 | 14 | YOLO26n/s/x, all RF-DETR, Grounding DINO, RT-DETR |
| `GridSample` | 7 | 123 | RF-DETR, RAFT, Grounding DINO, RT-DETR |
| `ReduceSum` | 7 | 57 | RF-DETR, RAFT, Grounding DINO, RT-DETR |
| `Sqrt` | 5 | 210 | RF-DETR family + RT-DETR; 48–60 nodes per model |
| `Cos`/`Sin` | 5 each | 49 each | RF-DETR family + Grounding DINO |
| `Not` | 5 | 46 | RF-DETR family + Grounding DINO |
| `Less` | 5 | 16 | RF-DETR family + Grounding DINO |
| `Exp` | 5 | 14 | RF-DETR family + YOLO26n-depth |
| `Mod` | 4 | 22 | YOLO26n/s/x + Grounding DINO |
| `Squeeze` | 3 | 530 | Grounding DINO, RT-DETR, FoundationPose scorer |
| `Pad` | 3 | 28 | RAFT, Grounding DINO, RT-DETR |
| `ReduceMean` | 3 | 27 | RT-DETR, FoundationPose scorer, ResNet `BR` fixture |
| `Clip` | 3 | 23 | Grounding DINO, YOLO26n-depth, RT-DETR |
| `Range` | 2 | 472 | RAFT, Grounding DINO |
| `ScatterND` | 2 | 132 | Grounding DINO, RF-DETR keypoint |
| `Identity` | 2 | 47 | Grounding DINO, RF-DETR keypoint |
| `InstanceNormalization` | 2 | 19 | RAFT, Grounding DINO |
| `Log` | 2 | 13 | Grounding DINO, RT-DETR |
| `Pow` | 2 | 13 | YOLO26n-depth, RT-DETR |

---

## F. Risks / defects for the release gate

1. **ResNet-18 fixture mismatch (critical).** The PASS evidence is for `/tmp/opencode/resnet18_export/resnet18.onnx` (`GlobalAveragePool`, 49 nodes), not `BR/native_parity_fixture/resnet18.onnx` (`ReduceMean`+`Reshape`, 49 nodes). `ReduceMean` is absent from the OpenCL table (`NF/src/opencl/table.cpp:61-99`), so the Neuriplo fixture would fail OpenCL with `no kernel for op 'ReduceMean'`. Pin the exact fixture in the release job or add the row; do not report both as the same artifact.
2. **`coverage.md:142` understates RT-DETR.** The RT-DETRv4-s ONNX exists locally at `NT/weights/rtv4_hgnetv2_s_model.onnx` (opset 16). It needs a 16→18 conversion and a gate; it is not a “no local export” family.
3. **Grounding DINO is a false OpenCL target.** Its 26 missing types are real, but the CPU/CUDA reference gate is blocked at `/model/model/Clip` on int64 inputs. No amount of OpenCL op coverage produces a PASS until an int64 Clip path exists (`coverage.md:159`; `opencl-kernel-matrix/validation.md:37`).
4. **Opset policy is a systemic export risk.** NERT accepts exactly opset 18 (`NF/src/model_loader.cpp:110`, `:1760-1784`). Exporter defaults are: YOLO/YOLO-pose **12**, RF-DETR/OWLv2/Grounding DINO/LGM **17**, ViTPose/Depth-Anything-V2/video **14**, RAFT **16**, RT-DETR **16**, torchvision classifier **unpinned**, ViT classifier **17**. Every new export needs an explicit `--opset 18` or a conversion step. The conversion sweep warns that `onnx.checker` does not prove semantic equivalence for down-converted graphs (`2026-10-18-opset18-conversions/validation.md:39-40`).
5. **Dynamic-shape blockers are not op-coverage blockers.** `FoundationPose ref2/refiner/rw/sw` and the deep-infer RF-DETR duplicate fail shape inference on dynamic `Slice` parameters (`coverage.md:140-141`; `opset18-conversions/validation.md:23`). No local graph uses an op type outside NERT’s loader allowlist (`NF/src/model_loader.cpp:884-955`); the failures are opset, dtype, or shape-parameter issues, not missing op names.
6. **Zero-extent / dynamic-shape handling.** RF-DETR’s transformer concatenates a zero-length slice; `Concat` accepts zero-length operands on CPU/CUDA/OpenCL (`coverage.md:235-238`). RAFT’s exporter default enables dynamic H/W; the actual gate export is static `[1,3,368,496]`. Video models are 5D with dynamic batch; Gaussian is dynamic batch. These need explicit static-shape fixtures for a reproducible gate.
7. **Postprocessing is not ONNX and is not covered by NERT gates.** OWLv2 and Grounding DINO require a host tokenizer (CLIP vocab/merges, BERT vocab) to turn text prompts into `input_ids`/`attention_mask`; the graph gate uses pre-tokenized fixtures (`coverage.md:117-127`; `NT/src/open_vocab_detection/open_vocab_detection_task.cpp:59-96`). Detection/segmentation postprocessors use host NMS (`yolo_postprocessor.cpp:413-436`), mask resize (`rfdetr_segmentation_postprocessor.cpp:134-137`), depth resize (`depth_anything_v2_postprocessor.cpp:79`), and RAFT flow resize (`raft_postprocessor.cpp:48-49`). Gaussian rendering/rasterisation is outside the library (`GaussianSplatting.md:40-45`). A NERT PASS says nothing about these host steps.
8. **Consumer device selection is narrower than the NERT matrix.** `neuriplo/backends/nert/src/NertInfer.cpp:81-87` selects only `CudaDevice` behind `-DNEURIPLO_NERT_CUDA=ON`, else `CpuDevice`; there is no `OpenCLDevice`/`HipDevice` selection (`coverage.md:209-215`). `neuriplo/docs/backends.yaml:227-241` records NERT as `gpu: true`, opt-in CUDA, experimental. OpenCL/HIP columns are NERT engine capability, not consumer-release capability. `requirements.md:37` explicitly puts AMD/Vulkan device layers out of scope, so HIP should not gate the Neuriplo release.
9. **RF-DETR provisioning is currently broken.** `import rfdetr` fails with `ImportError: cannot import name 'BackboneConfigMixin' from 'transformers'` (installed `transformers 4.35.0`; rfdetr expects a v5-era API). Existing RF-DETR ONNX artifacts are usable, but new sizes cannot be exported without a dependency upgrade/network.
10. **Gaussian splatting has a contract mismatch.** The LGM exporter runs `pipeline.unet` on a single `[1,3,256,256]` image (`export_lgm_to_onnx.py:52-92`), while `GaussianSplattingTask` requires 4 views (`gaussian_splatting_task.cpp:28-36`; `GaussianSplatting.md:37-40`). `diffusers` is not installed. GRM has no dedicated exporter.
11. **`--input_size` is parsed but not forwarded in the RF-DETR exporters.** `NT/export/detection/rfdetr/export.py:17,54-58` and `NT/export/segmentation/rfdetr/export.py:34,72-75` build `export_kwargs` without `input_size`; the docs’ `--input_size` is a no-op. The exported resolution comes from the RF-DETR class default (e.g., seg-small 384, keypoint 576), not the flag.
12. **Alias exactness is a trap.** The factory accepts broad `yolo*` but the task dispatcher special-cases exact tokens; `yolo26n` routes to `YOLO_STANDARD` instead of `YOLO_NMS_FREE` (`object_detection_task.cpp:187`). `rtdetrv4`, `rfdetr-nano`, `rfdetr-seg-small`, `raft-large`, `videomae-base...`, `vit-base...`, and `gemma-4-E4B...` are not accepted type tokens; use the family token and put the model ID/size in the model path.
13. **Duplicate and conflicting artifacts.** `OP/rfdetr_keypoint_preview-18.onnx` and `OP/rfdetr_keypoint_3542-18.onnx` are byte-size/node-identical duplicates (3,543 nodes, 2,355 non-Constant). `coverage.md:129-130` mentions a 1,392-node export that is not present locally. `OP/rfdetr_seg_small-18.onnx` (1,880 nodes) is the blocked deep-infer duplicate, not the gated 1,785-node file. `NT/model.onnx` is 0 bytes.
14. **Coverage counts are post-fold.** `coverage.md` node counts are NERT post-constant-folding; ONNX node counts differ by folded `Constant` nodes (e.g., Grounding DINO 31,828 ONNX vs 19,224 folded; RAFT 1,836 vs 1,299). Do not compare ONNX node totals directly with NERT dispatch counts.
</subagent>