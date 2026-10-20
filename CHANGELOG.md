# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Changed (breaking)

- Namespace `engine` → `nert` and public headers `include/engine/` →
  `include/nert/` (`#include "nert/engine.hpp"`). All C++ filenames are
  now lowercase `snake_case` (e.g. `src/model_loader.cpp`,
  `src/kernels/instance_normalization.cpp`); CTest suites `engine_*` →
  `nert_*`. No behavior change. Consumers (Neuriplo adapter) update
  includes + namespace; the `neuriplo_engine` CMake alias is kept.

### Added

- Opt-in CUDA device layer (`NERT_ENABLE_CUDA=OFF` by default): same model
  + same plan on CUDA — allocator, transfers, whole-graph staging, and an
  op-keyed table (cuBLAS `Gemm`, first-party `Relu`/`Add`, cuDNN `Conv`),
  matching the CPU oracle within tolerance. OFF link closure stays
  standard-library-only; CUDA/cuDNN link PRIVATE.
- RF-DETR norm-corridor CUDA rows: `Sqrt`, `Erf`, `ReduceSum` (float32
  double-accum + int64, incl. `noop_with_empty_axes` identity), and
  `LayerNormalization` (per-row double-accum, scalar/per-element
  Scale/B, optional Mean/InvStdDev) — 126 dispatch-gap nodes closed
  against the CPU oracle at the bench tolerance.
- R6 + Tier-5 CUDA rows (31 kernels): `Sin`/`Cos`/`Exp`/`Elu`/`Log`/
  `Floor`/`IsNaN`, `Abs`/`Neg`/`Pow`, `Greater`/`Less`/`GreaterOrEqual`/
  `LessOrEqual`/`And`/`Or`/`Not`, `Max`/`Min`/`Clip`,
  `Identity`/`Squeeze`/`Tile`/`Range`/`EyeLike`/`Pad`,
  `ReduceMean`/`ReduceL2`/`ArgMax`/`CumSum`/`ScatterND` — closing the
  R6a/R6b RF-DETR gaps and the Tier-5 open-vocab CUDA-missing set
  (GridSample and Einsum excluded) at the bench tolerance.
- CUDA closure: `GridSample` (first-party bilinear/nearest,
  zeros/border/reflection, align 0/1), `InstanceNormalization`
  (first-party per-(n,c) rows), `ConvTranspose` (cuDNN backward-data,
  `NERT_HAS_CUDNN` only, non-zero output_padding and asymmetric pads still
  rejected on CUDA `ConvTranspose`). Every CPU op now has a CUDA row (`ConvTranspose`
  only under `NERT_HAS_CUDNN`) — tables complete.
- OpenCL shared movement rows: `Transpose`, `Slice`, `Concat`,
  `Shape`, `Unsqueeze`, `Gather` (hand-written OpenCL C, oracle
  vs device green on the RTX 3060) — table 9→15; per-model portable
  gaps recomputed in the task-models coverage matrix.
- OpenCL pools: `MaxPool`, `AveragePool`, `GlobalAveragePool`
  (verified CUDA-row ports; ResNet-18 portable gap down to
  `Conv`/`Flatten`/`Gemm`) — table 15→18.
- RF-DETR seg-small CUDA gate: boxes 9.9e-04, labels 3.1e-04, top-1
  100%, binarized mask IoU min 0.9995 — detection-identical. The gate
  exposed two blockers, both fixed: `Concat` accepts zero-extent
  operands on every device, and the CUDA math modes now truly disable
  TF32 (`CUBLAS_PEDANTIC_MATH`/`CUDNN_FMA_MATH`; `DEFAULT_MATH` still
  allows it) after TF32 noise flipped near-tie TopK ordering. The
  coverage matrix's RAFT row was corrected: the local `ref.onnx` is a
  registration refiner, not torchvision RAFT, and no RAFT export is
  present locally.
- Opset-18 conversion sweep: 13 local exports now load and are CPU/CUDA
  op-complete — Grounding DINO (19,224 folded nodes), RF-DETR
  keypoint/seg-nano-576/nano-1101, YOLO26s/26x/26n-depth/11s/v8n-pose/
  26s-pose, and the FoundationPose scorer. Loader fixes: INT32 constants
  widen onto int64 lanes, and constant-parameter evaluation covers
  `Mod`, `Transpose` and single-axis N-D `Slice` (Grounding DINO's pad
  chain); regression test `PadWithEvaluatedParameterChain`.
- OpenCL `Flatten`/`Gemm`: a validated flat copy and a first-party GEMM
  kernel (transA/transB, alpha/beta, `[N]`/`[M,N]` bias) — table 18→20.
  ResNet-18's portable gap is now just `Conv`; YOLO/OWLv2/RF-DETR drop
  `Flatten`/`Gemm` as well.
- OpenCL `Conv`: first-party N-D cross-correlation (rank 3–5, groups,
  strides, dilations, explicit/SAME pads, bias) — table 20→21.
  ResNet-18 is op-complete on OpenCL (0 missing types); YOLO26n's gap
  drops to 12 types, OWLv2 to 16, RF-DETR to 25.
- OpenCL `MatMul` (1-D promotion, NumPy batch broadcast) and
  `LayerNormalization` (axis/epsilon, scale/bias, float accumulation) —
  table 21→23. YOLO26n drops to 11 types, OWLv2 to 14, RF-DETR to 23;
  RAFT-large's OpenCL gap is 10 types/122 nodes.
- OpenCL OWLv2 batch: Cast, ConstantOfShape, Equal, Greater,
  LessOrEqual, And, IsNaN, Erf, Elu, ReduceL2, ArgMax and the two target
  Einsum equations — table 25→37. **OWLv2 is op-complete and passes the
  full OpenCL gate** (logits exact, top-1 3600/3600, boxes max
  5.093e-04, 0 over tolerance, 4 s). YOLO26n drops to 6 types/21 nodes,
  RF-DETR to 14/84, RAFT to 7/116.
- Grounding DINO: the CPU/CUDA reference gate is blocked at
  `/model/model/Clip` — the export clips int64 token ids while the Clip
  row is float32-only. Op-type coverage did not catch this dtype variant;
  an ORT reference was generated and the blocker is recorded in the
  matrix.
- RAFT-large: real torchvision opset-18 export provisioned (1,836 ONNX
  nodes) and the full CPU+CUDA gate passes against ORT — flow maxdiff
  1.317e-04 (CPU) / 1.344e-04 (CUDA), tolerance 5.324e-03, on a real
  image pair. The earlier
  `ref.onnx` was a FoundationPose refiner, not RAFT.
- OpenCL full-model execution: handle-based devices now back each planned
  arena offset with its own buffer (`cl_mem` handles cannot be offset);
  two-output regression test. **ResNet-18 is the first full model to pass
  on OpenCL** — top-1 107 = ORT, logits maxdiff 3.695e-06 in 54 ms.
- OpenCL `Expand` (runtime target shape, identity copy + broadcast) and
  `Where` (bool condition, f32/i64 values, broadcast) — table 23→25.
  YOLO26n drops to 9 types/28 nodes, OWLv2 to 12/42, RF-DETR to 21/153;
  RAFT-large to 9/119.
- OpenCL catalog rows and P0 gates: 18 new rows (`Sqrt`, `Cos`, `Sin`,
  `Exp`, `Not`, `Less`, `ReduceSum` incl. int64, `ReduceMax`, `Split`,
  `GatherElements`, `TopK`, `Mod`, `Tile`, `Resize` dual-mode,
  `GridSample` align 0/1, `InstanceNormalization`, `Range`, `Pad`) —
  table 37→55. YOLO26n, RF-DETR seg-small and RAFT-large full OpenCL
  gates pass; int64 `Clip` on CPU/CUDA unblocks Grounding DINO.
    - YOLO26n: 421 nodes dispatched, output0 [1,300,6], maxdiff
      4.8828e-04 (mean 1.4135e-05), 0/1800 over 1e-3*(1+|ref|), 6/6
      confident rows with identical classes, min IoU 0.99999840,
      837 ms.
    - RF-DETR seg-small: 1,255 folded nodes dispatched, dets maxdiff
      6.822e-04 (0/400 over tolerance), labels top-1 100/100, mask IoU
      at logit>=0 min 0.999668655 / mean 0.999996687, 2.27 s.
    - RAFT-large: 1,299 folded nodes dispatched, flow maxdiff
      1.3232e-04 (mean 1.5064e-05), 0/365,056 over 5.324e-03,
      OpenCL vs CPU 1.2832e-04, 4m17s.
- OpenCL catalog completion: `Clip` (float32/int64), `Log`, `Pow`,
  `Max`, `Min`, `Or`, `Identity`, `Squeeze`, `CumSum`, `EyeLike`,
  `ScatterND`, `ReduceMean` and `ConvTranspose` — final row set 68
  rows (table 55→68, verified against `src/opencl/table.cpp`):
  first-party `Relu`/`Add`/`Mul`/`Sub`/`Div`/`Sigmoid`/`Tanh`/`Softmax`/
  `Reshape`, movement `Transpose`/`Gather`/`Slice`/`Concat`/`Shape`/
  `Unsqueeze`, pools `MaxPool`/`AveragePool`/`GlobalAveragePool`,
  `Flatten`/`Gemm`/`Conv`/`MatMul`/`LayerNormalization`, `Expand`/
  `Where`, the OWLv2 batch `Cast`/`ConstantOfShape`/`Equal`/`Greater`/
  `LessOrEqual`/`And`/`IsNaN`/`Erf`/`Elu`/`ReduceL2`/`ArgMax`/`Einsum`,
  the P0 batch `Sqrt`/`Cos`/`Sin`/`Exp`/`Not`/`Less`/`ReduceSum`/
  `ReduceMax`/`Split`/`GatherElements`/`TopK`/`Mod`/`Tile`/`Resize`/
  `GridSample`/`InstanceNormalization`/`Range`/`Pad`, and the
  completion batch listed above. Every loadable catalog model now has
  0 missing OpenCL op types, including Grounding DINO, RF-DETR
  keypoint, YOLO26n-depth, RT-DETRv4-s and the FoundationPose scorer.
  OpenCL suite 28/28, format clean. Full-model gates recorded
  2026-10-19 against ORT (tolerance `1e-3*(1+|ref|)`):
    - YOLO11s: 321 folded nodes; 0 elements over tolerance on CPU,
      OpenCL and CUDA; max abs CPU 9.77e-04, OpenCL 1.92e-03, CUDA
      1.97e-03; OpenCL 436 ms, CUDA 1.4 s, CPU 119 s.
    - YOLOv8n-pose: 267 nodes; 0 over tolerance on all three devices;
      max abs CPU 4.88e-04, OpenCL 9.61e-04, CUDA 1.31e-03.
    - YOLO26s-pose: 401 folded nodes; 0 over tolerance on all three
      devices; max abs CPU 4.88e-04, OpenCL 8.70e-04, CUDA 1.50e-03;
      same top-38 anchors as ORT, box and keypoint error <= 1.9e-04 px.
    - RF-DETR nano-1101: 1,002 folded nodes; dets max abs OpenCL
      5.44e-05, CUDA 1.25e-05; labels max abs OpenCL 3.48e-04, CUDA
      2.50e-04; labels top-1 300/300 on all devices; 0 over tolerance.
    - YOLO26n-depth: 328 folded nodes; 0 over tolerance on all three
      devices; max abs CPU 1.34e-05, OpenCL 4.20e-05, CUDA 1.72e-05;
      `ConvTranspose` dispatched on every device.
    - RF-DETR keypoint-preview: 2,355 folded nodes; dets max abs
      OpenCL 1.34e-05; labels max abs OpenCL 2.98e-05, top-1 100/100
      on all devices; keypoints max abs OpenCL 2.13e-03, max tolerance
      ratio 0.197; 0 over tolerance; OpenCL `Identity` x3 and
      `ScatterND` x17 dispatched.
    - YOLO26s / YOLO26x: PASS with the R-6 deviation — 10/1800
      elements over tolerance in the sub-0.001 confidence tail, a TopK
      tie flip between two candidates ~3e-8 apart that reproduces
      identically on the CPU control; confident rows identical (7/7
      and 5/5, score max abs 5.6e-06 / 1.2e-07, box IoU >= 0.999999).
    - RF-DETR seg-nano-576: CPU matches ORT (dets max 2.10e-05, labels
      top-1 100/100, mask IoU min 0.99980); OpenCL and CUDA fail on
      ONE TopK near-tie at rank 99 (candidate scores tied to 1.43e-06
      in the reference vs fp32 reduction-order noise 4.82e-05 on the
      same score vector); with the TopK selection pinned to ORT's, all
      three devices reach 0 over tolerance — blocker owned by the
      export/topology, not a kernel row.
    - RT-DETRv4-s: CPU and OpenCL pass (boxes max abs CPU 7.63e-04,
      OpenCL 1.50e-03; scores 1.05e-05 / 3.83e-05; labels int64
      identical 300/300); CUDA is a GAP — the graph's one
      asymmetric-pad `Conv` (`/model/backbone/stem/stem2b/conv/Conv`,
      pads [0,0,1,1]) is rejected by the CUDA frontend (cuDNN requires
      symmetric spatial padding).
    - Grounding DINO: the previous int64 `Clip` BLOCKED reason is gone;
      the CPU and CUDA full-model runs pass and the ORT comparison is
      recorded; the OpenCL leg is a backend gap. All 26 op types
      the model uses are present in the 68-row OpenCL table.
- CUDA `Conv` supports asymmetric padding via an internal explicit zero
  pre-pad in both the legacy and the cuDNN-frontend rows (the input is
  copied at spatial offset `pad_begin` into a zero-filled temporary and
  cuDNN then runs with pad=0); the symmetric path is unchanged. Negative
  pads are rejected with a named error. RT-DETRv4-s's full-model CUDA
  gate now passes — boxes 0/1200 (max 7.93e-04), scores 0/300 (max
  2.46e-05), labels 300/300 exact, 11 detections over 0.25.
- Grounding DINO task-equivalence gate: CPU and CUDA pass — `pred_boxes`
  0/3600 over `1e-3(1+|ref|)` (max 7.00e-05 / 7.03e-05), per-box IoU
  min 0.99973 (900/900), top-1 900/900 after canonicalizing the TopK
  near-tie permutation, decoded detection set identical at the 0.25/0.25
  Hugging Face proxy (1 kept query, margin 1.39e-02). The OpenCL leg is a
  backend gap: the handle path demands 11,028.5 MiB on a 5,803.3 MiB device
  against a 1,687.9 MiB live set.
- RF-DETR seg-nano-576: PASS with the documented rank-99 TopK near-tie
  deviation — ORT selects 182 and the GPUs 1503 on a 1.43e-06 tie against
  4.82e-05 of fp32 reduction-order noise; with the selection pinned to
  ORT's (a single-input counterfactual, ORT-neutral) all three devices give
  `dets` 0/400, `labels` 0/9,100, top-1 100/100, and mask IoU min 0.999800
  at `raw>=0` (CUDA 0.984375 at `raw>=0.5`). No engine change was made or
  is authorised.
- Tranche-1 export-first batch: four catalog entries newly gated on CPU,
  CUDA and OpenCL at opset 18 — YOLOv8n detector (entry 17; 301 nodes on
  disk / 260 dispatched), YOLO11m-seg (entry 18; 515 / 467, both outputs
  clean), OWL-ViT (entry 25; 1,418 / 1,071, both outputs clean) and
  ResNet-50 (entry 30; 122 / 122), every device exit 0 and 0 elements over
  the scaled tolerance. YOLO11m-seg was exported from the local checkpoint;
  YOLOv8n and OWL-ViT were exported at opset 18; ResNet-50 was converted
  from an opset-20 raw export by `onnx.version_converter` — conversion does
  not prove semantic equivalence — the full-model gate is the equivalence
  evidence. OWL-ViT used a worker-authored exporter (the repository has
  none) and was re-exported at two sites in the exporter only to remove a
  DOUBLE `Cast` and a constant `Reciprocal` the loader rejects; the patched
  artifact was proven bit-identical to the rejected one on both outputs
  across five input sets. No kernel row was added: the tables are unchanged
  at CPU 72 / CUDA 72 / OpenCL 68 / HIP 9. Scoreboard moved 15 → 19 and the
  release is still blocked by 1 open entry (Grounding DINO) and 10 missing
  exports. No release was cut.
- Wave-A tranche (boundary 4): ViTPose-base-simple (entry 26) exported at
  opset 18 by the Neuriplo ViTPose exporter (536 nodes on disk / 422
  dispatched — the 114 `Constant` nodes are folded at load) and gated on CPU,
  CUDA and OpenCL. Every leg exited 0 with 422 dispatched, and 0 elements
  were over the `--tol 1e-3` scaled tolerance on `heatmaps [1,17,64,48]`;
  export sha256 `b72b99e52da700dbbef81c7dd5e677b40002338724dd7ec7430cb4ba84fee5c8`.
  Export note: torch 2.12's `torch.onnx.export` defaults to `dynamo=True` and
  the stock run produced a non-self-contained artifact, so the exporter was
  copied to `/tmp` and `dynamo=False` added at the call site only; the system
  `transformers` 4.35.0 does not implement ViTPose, so 4.57.1 was installed
  in an isolated venv under the export directory — the system site-packages
  were not modified. No kernel row was added (tables unchanged at
  CPU 72 / CUDA 72 / OpenCL 68 / HIP 9) and no engine source file changed.
  Scoreboard moved 19 → 20 and the missing-export count 10 → 9. No release
  was cut.

### Fixed

- OpenCL `ReduceSum` int64 path, found by the RF-DETR seg-small gate:
  `/transformer/ReduceSum` (bool→Cast int64 data, axes [-1],
  keepdims 1) threw "data tensors must be float32"; `ReduceSum` now
  plans float32 or int64 data (output dtype matching) and launches a
  dedicated `ReduceSum/int64` kernel.
- int64 `Clip` dtype preservation in shape inference: the `Clip` output
  now keeps the data input's dtype (float32 or int64) instead of
  coercing, and the CPU/CUDA `Clip` rows accept int64 — unblocking
  Grounding DINO's reference gate at `/model/model/Clip`; the gate
  re-run has since passed on CPU and CUDA under the task-equivalence bar, with
  the OpenCL leg a backend gap).
- Review-driven OpenCL corrections landed after `b905792`: `ScatterND`
  now applies duplicate update slices in order (last duplicate wins,
  matching the CPU oracle), `Or` enforces the oracle's two-input
  arity, `ReduceMean`'s `noop_with_empty_axes` now applies only when an
  axes input is supplied (shape inference aligned with the CPU
  kernel), and the rank cap was removed from the `Identity` and
  `Squeeze` copy rows.

## [0.1.0] - Unreleased

### Added
- First-party ONNX inference runtime (NERT, the Neuriplo Engine Runtime):
  model loader, shape inference, single-arena memory planner, constant
  folding, device seam, and executor, targeting ONNX opset 18.
- CPU reference kernels for the operator set needed by the YOLO families, ViT,
  ViTPose, Depth Anything V2, VideoMAE, ViViT, TimeSformer, and the DETR
  family (RT-DETR v4, D-FINE/DEIM, DEIMv2, RF-DETR, RF-DETR-Seg).
- Tier-5 open-vocabulary kernels: `Elu`, `Max`, `Min`, `LessOrEqual`,
  `GreaterOrEqual`, `Or`, `ReduceL2`, `ArgMax`, `CumSum`, `EyeLike`,
  `ScatterND` — each with shape rule, loader allowlist entry + opset-18
  attribute policy, and hand-computed unit tests incl. off-default
  attributes and a negative load/shape case. RAFT `InstanceNormalization`
  (Tier-4) shipped earlier with RAFT parity `max_scaled_diff=6.10e-05`
  (atol 1e-3).
- float32 compute with int64/bool shape data and I/O; no dependencies beyond
  the C++ standard library.
- Standalone CMake build, install rules, and `find_package(nert)` package
  config exporting `nert::nert`.
- GoogleTest suites (`engine_*`) driven by a ResNet-18 fixture provisioned with
  `tools/provision_resnet18.py`.
- Opt-in `bench/` kernel measurement harness (`NERT_BUILD_BENCH=OFF` by
  default), mirroring the gpu-playground per-kernel layout (`main.cpp` +
  `cuda/` + `opencl/` + `opencl_cpp/` + vendored `.hpp` timing helpers):
  starting with `gemm` on CUDA and OpenCL C/C++ against a bench-local CPU
  oracle. Tooling only; the runtime stays standard-library-only.
