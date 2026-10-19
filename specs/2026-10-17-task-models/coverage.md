# Coverage matrix — task models × devices

Date: 2026-10-18; updated 2026-10-19 with the P0 OpenCL catalog batch
(kernel table 37 → 55) and the YOLO26n / RF-DETR seg-small / RAFT-large
OpenCL gates. Measured pass against the working tree. The CUDA
closure slice, the OpenCL movement rows (Transpose/Slice/Concat/Shape/
Unsqueeze/Gather) and the OpenCL pools (MaxPool/AveragePool/
GlobalAveragePool) all landed the same day and are folded in here.
This is the living matrix required by [`requirements.md`](requirements.md)
(in-scope coverage tracking) and [`plan.md`](plan.md) Phase 4. Update it
whenever a full-model gate runs or a device kernel row lands; do not let
a missing check become a silent hole.

## How to read this

- **PASS** — a full-model run against the reference was recorded in a
  packet's `validation.md`.
- **OPS** — every op type in the NERT-loaded graph has a row on that
  device; the full-model gate has not been run. This is a kernel-coverage
  statement, not a correctness claim.
- **GAP(n)** — `n` op types used by the model have no row on that device.
  Counts in parentheses are NERT post-constant-folding node touches.
- **CPU parity** — the model ran on CPU against ONNX Runtime and the
  numbers are in the cited packet.
- **Recorded** — a model-zoo or roadmap result exists, but it is not a
  packet-level full-model gate.
- **BLOCKED** — no export NERT can load (opset or missing export), so no
  NERT gate is possible yet.
- **NO** (Neuriplo column) — the consumer adapter cannot request GPU. It
  is a separate layer from NERT kernel coverage.

NERT node counts are post constant-folding, which is what the executor
dispatches. ONNX file node counts are given for reference; the difference
is folded `Constant` nodes.

## Kernel-table baseline

| Layer | Op rows | Notes |
| --- | ---: | --- |
| CPU (oracle) | 72 | Reference set; every inventoried model's op types are covered after constant folding. |
| CUDA | 72 | Identical set to CPU. `Conv` and `ConvTranspose` are cuDNN-gated; a CUDA build without cuDNN lacks those two rows. Strict float32: cuBLAS runs `CUBLAS_PEDANTIC_MATH` and cuDNN `CUDNN_FMA_MATH`, so TF32 is off on Ampere+. |
| OpenCL | 68 | `Add`, `Div`, `Mul`, `Relu`, `Reshape`, `Sigmoid`, `Softmax`, `Sub`, `Tanh`, `Transpose`, `Slice`, `Concat`, `Shape`, `Unsqueeze`, `Gather`, `MaxPool`, `AveragePool`, `GlobalAveragePool`, `Flatten`, `Gemm`, `Conv`, `MatMul`, `LayerNormalization`, `Expand`, `Where`, `Cast`, `ConstantOfShape`, `Equal`, `Greater`, `LessOrEqual`, `And`, `IsNaN`, `Erf`, `Elu`, `ReduceL2`, `ArgMax`, `Einsum`, `Sqrt`, `Cos`, `Sin`, `Exp`, `Not`, `Less`, `ReduceSum`, `ReduceMax`, `Split`, `GatherElements`, `TopK`, `Mod`, `Tile`, `Resize`, `GridSample`, `InstanceNormalization`, `Range`, `Pad`, `Clip`, `Log`, `Pow`, `Max`, `Min`, `Or`, `Identity`, `Squeeze`, `CumSum`, `EyeLike`, `ScatterND`, `ReduceMean`, `ConvTranspose` (2026-10-19 P0 batch: 37 → 55; catalog completion: 55 → 68). |
| HIP | 9 | `Add`, `Div`, `Mul`, `Relu`, `Reshape`, `Sigmoid`, `Softmax`, `Sub`, `Tanh`. Lags OpenCL by the movement six and the three pools. |

The CUDA table being op-complete is the key result: **there are no
missing CUDA op types to add for any current model.** The OpenCL table
now also covers every op type in the loadable catalog (68 rows). The
remaining portable risk is dtype/shape variants, full-model gates for
converted variants, and HIP, which still has only the original nine
rows.

## Models NERT can load (opset 18)

| Model | Task | NERT nodes (ONNX) | CPU | CUDA | OpenCL | HIP | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| ResNet-18 | classification | 49 (49) | PASS | PASS | PASS | GAP(5) / 24 | CUDA gate V-1; OpenCL gate 2026-10-19: top-1 107 = ORT, logits maxdiff 3.695e-06. |
| YOLO26n | detection | 421 (485) | PASS | PASS | PASS | GAP(21) / 197 | CUDA gate R-6; 6/6 confident rows. OpenCL gate 2026-10-19: 421 nodes dispatched, output0 [1,300,6], maxdiff 4.8828e-04 (mean 1.4135e-05), 0/1800 over tolerance, 6/6 confident rows, classes [5,0,0,0,0,0] identical, min IoU 0.99999840, 837 ms; CPU control reproduced the R-6 output bit-identically. |
| OWLv2-base-patch16 | open-vocab detection | 952 (1219) | PASS | PASS | PASS | GAP(24) / 453 | CPU/CUDA V-3/V-3b; OpenCL gate 2026-10-19: logits exact, top-1 3600/3600, boxes max 5.093e-04. |
| RF-DETR seg-small | detection + segmentation | 1255 (1785) | PASS | PASS | PASS | GAP(33) / 701 | CUDA gate 2026-10-18: boxes 9.9e-04, labels 3.1e-04, top-1 100%, mask IoU 0.9995. OpenCL gate 2026-10-19: 1,255 folded nodes dispatched, dets maxdiff 6.822e-04 (0/400 over tolerance), labels top-1 100/100 (maxdiff 2.904e-04), mask IoU at logit>=0 min 0.999668655 / mean 0.999996687, 2.27 s. |
| RAFT-large (torchvision) | optical flow | 1299 (1836) | PASS | PASS | PASS | GAP(19) / 501 | Real opset-18 export 2026-10-19; CPU max 1.317e-04, CUDA max 1.344e-04 vs ORT (tol 5.324e-03). OpenCL gate 2026-10-19: 1,299 folded nodes dispatched, flow maxdiff 1.3232e-04 (mean 1.5064e-05), 0/365,056 over 5.324e-03, 4m17s. |

RAFT-large was re-provisioned as a real torchvision export (opset 18, 1,836 ONNX nodes, 1,299 after folding) on 2026-10-19. The earlier 156-node row came from `/tmp/opencode/ref.onnx`, a FoundationPose-style registration refiner (inputs `A`/`B` 6×160×160, outputs `trans`/`rot`), not RAFT; that misattribution is corrected here. Both CPU and CUDA full-model gates use the new export (1.317e-04 / 1.344e-04 vs ORT).

### Missing OpenCL rows by model

Counts are NERT post-fold node touches. HIP is a subset of this table
plus the OpenCL-only movement/pool rows; its exact gaps follow.

The P0 batch (`Sqrt` … `Pad`, 18 rows) and the catalog-completion batch
(`Clip`, `Log`, `Pow`, `Max`, `Min`, `Or`, `Identity`, `Squeeze`,
`CumSum`, `EyeLike`, `ScatterND`, `ReduceMean`, `ConvTranspose`,
13 rows) closed every OpenCL op-type gap in the loadable catalog.
Counts are recomputed against the current 68-row `src/opencl/table.cpp`
from the inventory census (post-fold op types), not re-probed. A zero
gap is OPS, not PASS; full-model gates are still pending where noted.

- **YOLO26n — 0 types.** Op-complete and full-model gated on OpenCL
  (2026-10-19). Former gap: `Split=12`, `GatherElements=3`,
  `Resize=2`, `TopK=2`, `Mod=1`, `ReduceMax=1` (6 types, 21 nodes).
- **OWLv2 — 0 types.** Op-complete on OpenCL and full-model gated.
- **RF-DETR seg-small — 0 types.** Op-complete and full-model gated on
  OpenCL (2026-10-19). Former gap: 14 types, 84 nodes (`Sqrt=48`,
  `ReduceSum=5`, `Split=5`, `Cos=4`, `GridSample=4`, `Sin=4`,
  `Exp=3`, `Not=3`, `Tile=3`, `GatherElements=1`, `Less=1`,
  `ReduceMax=1`, `Resize=1`, `TopK=1`); the gate also surfaced the
  int64 `ReduceSum` fix (`/transformer/ReduceSum`, bool→Cast int64,
  axes [-1], keepdims 1).
- **RAFT-large — 0 types.** Op-complete and full-model gated on OpenCL
  (2026-10-19). Former gap: `GridSample=48`, `Split=48`,
  `InstanceNormalization=15`, `Range=2`, `Pad=1`, `ReduceSum=1`,
  `Tile=1` (7 types, 116 nodes).
- **YOLO26s / YOLO26x / YOLO11s / YOLOv8n-pose / YOLO26s-pose — 0
  types** (recomputed; full-model gates pending). Their former gaps
  were subsets of `Split`, `Resize`, `GatherElements`, `TopK`,
  `ReduceMax`, `Mod` and `Tile`, all implemented.
- **YOLO26n-depth — 0 types.** Former gap: `Clip=1`, `ConvTranspose=1`,
  `Pow=1`; all three rows landed 2026-10-19. Full-model gate pending.
- **RF-DETR keypoint-preview — 0 types.** Former gap: `Identity=3`,
  `ScatterND=17`; both rows landed 2026-10-19. Full-model gate pending.
  **RF-DETR seg-nano-576 / nano-1101 — 0 types** (recomputed from
  14/83 and 13/71; gates pending).
- **Grounding DINO — 0 types.** Former gap: 10 types/1,217 nodes
  (`Squeeze=513`, `Max=255`, `Min=255`, `ScatterND=115`, `Identity=44`,
  `Clip=13`, `Log=12`, `CumSum=8`, `Or=1`, `EyeLike=1`). The int64
  `Clip` CPU/CUDA rows, the shape-inference dtype preservation, and the
  OpenCL completion rows landed 2026-10-19. Reference gate re-run
  pending; no PASS is claimed.
- **RT-DETRv4-s — 0 types** after the 16→18 conversion. Former gap:
  5 types/63 nodes (`Squeeze=16`, `ReduceMean=25`, `Pow=12`, `Clip=9`,
  `Log=1`); all rows landed 2026-10-19. Full-model gate pending.
- **FoundationPose scorer (out-of-scope) — 0 types.** Former gap:
  `ReduceMean=1`, `Squeeze=1`; both rows landed 2026-10-19.

### Missing HIP rows by model

HIP carries only the original nine rows, so its per-model gap is every
op type outside that set. The full primary-model lists are:

- **ResNet-18 — 5 types, 24 nodes:** `Conv=20`, `Flatten=1`, `Gemm=1`, `GlobalAveragePool=1`, `MaxPool=1`.
- **YOLO26n — 21 types, 197 nodes:** `Conv=102`, `Concat=28`, `Split=12`, `Unsqueeze=11`, `Expand=8`, `Transpose=6`, `MatMul=4`, `Cast=3`, `GatherElements=3`, `MaxPool=3`, `ConstantOfShape=2`, `Equal=2`, `Resize=2`, `Slice=2`, `TopK=2`, `Where=2`, `Flatten=1`, `Gather=1`, `Mod=1`, `ReduceMax=1`, `Shape=1`.
- **OWLv2 — 24 types, 453 nodes:** `MatMul=202`, `Transpose=97`, `LayerNormalization=53`, `Where=28`, `IsNaN=12`, `Gather=11`, `Cast=8`, `Concat=5`, `Shape=5`, `ConstantOfShape=4`, `Equal=4`, `Erf=4`, `Expand=3`, `ReduceL2=3`, `Slice=3`, `And=2`, `Flatten=2`, `ArgMax=1`, `Conv=1`, `Einsum=1`, `Elu=1`, `Greater=1`, `LessOrEqual=1`, `Unsqueeze=1`.
- **RF-DETR seg-small — 33 types, 701 nodes:** `MatMul=158`, `Transpose=133`, `Slice=72`, `LayerNormalization=56`, `Sqrt=48`, `Cast=38`, `Concat=38`, `Shape=32`, `Unsqueeze=23`, `Erf=17`, `Conv=14`, `Expand=8`, `ConstantOfShape=6`, `Where=6`, `ReduceSum=5`, `Split=5`, `Cos=4`, `Equal=4`, `Gather=4`, `Gemm=4`, `GridSample=4`, `Sin=4`, `Exp=3`, `Not=3`, `Tile=3`, `Greater=2`, `And=1`, `Einsum=1`, `GatherElements=1`, `Less=1`, `ReduceMax=1`, `Resize=1`, `TopK=1`.
- **RAFT-large — 19 types, 501 nodes:** `Conv=190`, `Concat=135`, `GridSample=48`, `Split=48`, `Transpose=27`, `InstanceNormalization=15`, `Gather=7`, `Unsqueeze=7`, `Shape=5`, `Slice=4`, `AveragePool=3`, `Expand=3`, `Cast=2`, `Range=2`, `ConstantOfShape=1`, `MatMul=1`, `Pad=1`, `ReduceSum=1`, `Tile=1`.

No HIP full-model gate is claimed; the ROCm toolchain and AMD hardware
are unavailable, so HIP remains unsupported for this release.

## Converted exports now loadable (opset-18 conversion, 2026-10-18)

`onnx.version_converter.convert_version(model, 18)` plus
`onnx.checker`; each row then loaded through NERT's own loader and
shape inference. **CPU and CUDA op-type coverage is complete for every
row** (0 missing types), but op-type counts hide dtype variants:
Grounding DINO clips int64 token ids (`/model/model/Clip`); the int64
`Clip` rows and the shape-inference dtype preservation landed
2026-10-19, unblocking its reference gate (re-run pending). The other
variants have no reference gate yet — they are OPS, not PASS. OpenCL/HIP
gaps are ONNX op-type counts (Constant excluded). The OpenCL column was
recomputed 2026-10-19 against the current 68-row OpenCL table after the
catalog-completion batch; the HIP column still reflects the 2026-10-18 census.

| Model | Source opset | NERT nodes | OpenCL gap | HIP gap |
| --- | ---: | ---: | --- | --- |
| Grounding DINO | 17 | 19,224 | 0 | 45 / 15,336 |
| RF-DETR keypoint-preview | 17 | 2,355 | 0 | 33 / 1,486 |
| RF-DETR seg-nano-576 | 17 | 1,216 | 0 | 33 / 675 |
| RF-DETR nano-1101 | 17 | 1,002 | 0 | 31 / 567 |
| FoundationPose scorer | 17 | 145 | 0 | 10 / 98 |
| YOLO26s | 20 | 391 | 0 | 17 / 176 |
| YOLO26x | 19 | 606 | 0 | 17 / 247 |
| YOLO26n-depth | 20 | 328 | 0 | 11 / 139 |
| YOLOv8n-pose | 20 | 267 | 0 | 7 / 110 |
| YOLO11s | 12 | 321 | 0 | 8 / 132 |
| YOLO26s-pose | 12 | 401 | 0 | 8 / 164 |

The two RF-DETR keypoint-preview exports (1,392 and 3,542 ONNX nodes)
fold to the same 2,355-node graph; the table lists it once. The
FoundationPose `ref` and `scorer` components load; `ref2`/`refiner`/
`rw`/`sw` remain blocked (below).

## Exports NERT still cannot load

Blockers are per-graph now, not the opset:

| Export/file | Task | Opset | ONNX nodes | Blocker |
| --- | --- | ---: | ---: | --- |
| FoundationPose `ref2`, `refiner`, `rw`, `sw` | 6-D pose | 17→18 | 363 / 382 / 377 / 300 | Converted; shape inference needs dynamic `Slice` parameters (attention reshape). |
| RF-DETR seg-small (deep-infer duplicate) | detection + segmentation | 17→18 | 1,879 | Converted; dynamic `Slice` parameters; the NERT-side export already loads. |
| DETR family, ViT, ViTPose, Depth-Anything-V2, VideoMAE, ViViT, TimeSformer | various | — | — | No local export; CPU coverage recorded in the roadmap. |

RAFT is now provisioned as a real torchvision export and is in the models table above. Grounding DINO needed two loader fixes: INT32 constants (folded onto
int64 lanes, matching the Cast emulation) and constant-parameter
evaluation through `Mod`, `Transpose`, and single-axis N-D `Slice`
(its pad chain). Both landed 2026-10-18; the graph loads at 19,224
folded nodes.

## Full-model gates

| Model | CPU | CUDA | OpenCL | HIP | Evidence / blocker |
| --- | --- | --- | --- | --- | --- |
| ResNet-18 | PASS | PASS | PASS | GAP(5/24) | V-1; OpenCL gate 2026-10-19: top-1 107 = ORT, logits maxdiff 3.695e-06 (mean 7.944e-07), 54 ms. |
| YOLO26n | PASS | PASS | PASS | GAP(21/197) | R-6 (6/6 confident rows, min IoU 0.9998039; detection-identical). OpenCL gate 2026-10-19: maxdiff 4.8828e-04 (mean 1.4135e-05), 0/1800 over 1e-3*(1+|ref|), 6/6 confident rows, classes [5,0,0,0,0,0] identical, min IoU 0.99999840, 837 ms; CPU control reproduced the R-6 output bit-identically. |
| OWLv2 | PASS | PASS | PASS | GAP(24/453) | V-3/V-3b; OpenCL gate 2026-10-19: logits exact, top-1 3600/3600, boxes max 5.093e-04, 0 over tolerance, 4 s. |
| RF-DETR seg-small | PASS | PASS | PASS | GAP(33/701) | CUDA gate boxes 9.938e-04, labels 3.080e-04, top-1 100%, mask IoU 0.999528. OpenCL gate 2026-10-19: dets maxdiff 6.822e-04 (0/400 over tolerance), labels top-1 100/100 (maxdiff 2.904e-04), mask IoU at logit>=0 min 0.999668655 / mean 0.999996687, 2.27 s; the gate exposed and fixed the int64 `ReduceSum` row. |
| RAFT-large | PASS | PASS | PASS | GAP(19/501) | Full gate 2026-10-19 on a real `bus.jpg` pair: CPU maxdiff 1.317e-04 (1,074 s), CUDA 1.344e-04; tolerance 5.324e-03. OpenCL gate 2026-10-19: flow vs `flow_ort_real.npy` maxdiff 1.3232e-04 (mean 1.5064e-05), 0/365,056 over 5.324e-03, max ratio 1.3162e-04; OpenCL vs CPU 1.2832e-04; 4m17s. |
| Grounding DINO | BLOCKED | BLOCKED | GAP(10/1,217) | GAP(45/15,336) | Loads (19,224 folded nodes), but both CPU and CUDA fail at `/model/model/Clip`: int64 token ids with a float32-only Clip row. ORT reference generated; gate blocked on an int64 Clip variant. |
| Converted YOLO/RF-DETR variants | OPS | OPS | per converted table | per converted table | Load + shape inference only; no reference gate yet. |
| DETR/ViT/ViTPose/Depth-Anything/VideoMAE/ViViT/TimeSformer | BLOCKED | BLOCKED | BLOCKED | BLOCKED | No local export (searched 2026-10-18). |

### P0 portable gate result (2026-10-19)

The P0 OpenCL catalog batch (classes A/B/C: math unary + reduction
rows, index/move/integer rows, then resampling/normalization/range/pad
rows) took the kernel table 37 → 55 and closed the entire OpenCL gap of
all three P0 models: **YOLO26n, RF-DETR seg-small and RAFT-large pass
the full-model OpenCL gate** against ORT (evidence in the tables
above), joining ResNet-18 and OWLv2. Commits were pending at recording
time. The RF-DETR gate exposed and fixed the int64 `ReduceSum` OpenCL
row, and the int64 `Clip` CPU/CUDA rows plus shape-inference dtype
preservation unblock Grounding DINO's blocked reference gate.

## Portable-backend priority (OpenCL)

Updated 2026-10-19 after the P0 batch: every shared row the loadable
models needed has landed, so the primary models and the YOLO/RF-DETR
variants have no remaining OpenCL gap. The former priority list
(`Split`, `GatherElements`, `Resize`, `TopK`, `ReduceMax`, `Sqrt`,
`GridSample`, plus the one-model rows `ReduceSum`, `Cos`, `Sin`, `Exp`,
`Not`, `Tile`, `Less`, `Mod`, `InstanceNormalization`, `Range`, `Pad`)
is fully implemented. HIP additionally lacks the movement six and the
pools, so its unlock sets are unchanged and remain the larger ones
recorded above.

Model-centric milestones (OpenCL gaps after the P0 batch; HIP
unchanged):

| Model unlock | OpenCL | HIP |
| --- | ---: | ---: |
| ResNet-18 | 0 | 5 |
| OWLv2 | 0 | 24 |
| RAFT-large | 0 | 19 |
| YOLO26n | 0 | 21 |
| RF-DETR seg-small | 0 | 33 |

All five primary models are op-complete and gated on OpenCL. The
remaining OpenCL rows target Grounding DINO, RT-DETRv4, RF-DETR
keypoint (`Identity`/`ScatterND`) and YOLO26n-depth
(`Clip`/`ConvTranspose`/`Pow`), dominated by `Squeeze` (513 touches in
Grounding DINO alone), `Max`/`Min` (255 each), `ScatterND` (115),
`Identity`, `Pow`, `ReduceMean`, `Log`, `CumSum`, `Clip`, `EyeLike`
and `Or`.

## Non-kernel blockers

1. **Opset policy.** NERT loads opset 18 only. The 2026-10-18 sweep
   converted 17 candidates; 13 now load and are CPU/CUDA op-complete.
   FoundationPose `ref2`/`refiner`/`rw`/`sw` and the deep-infer RF-DETR
   duplicate still need dynamic `Slice` shape inference. RAFT and the
   DETR/ViT families have no local export.
2. **Consumer adapter.** Neuriplo's NERT backend selects
   `nert::CudaDevice()` behind `-DNEURIPLO_NERT_CUDA=ON` (adapter commit
   `014fe18`, 2026-10-18); the CUDA-leg full-model run is still pending,
   and there is no `OpenCLDevice`/`HipDevice` selection or GPU CI leg.
   `neuriplo/docs/backends.yaml` now records NERT as `gpu: true`
   (build-time opt-in). NERT kernel coverage alone does not change the
   consumer matrix.
3. **Pending gates.** ResNet-18, OWLv2, YOLO26n, RF-DETR seg-small and
   RAFT-large pass CPU/CUDA/OpenCL (2026-10-19). Remaining OpenCL gaps:
   Grounding DINO (10 types), RT-DETRv4 (5, after conversion), RF-DETR
   keypoint (2), YOLO26n-depth (3), FoundationPose scorer (2); HIP
   passes none (gap tables above).
4. **Grounding DINO.** Converted and loading (19,224 folded nodes; all
   CPU/CUDA op **types** covered), but the CPU/CUDA reference gate is
   BLOCKED at `/model/model/Clip`: the export clips int64 token ids and
   the Clip row is float32-only. An int64 Clip variant is required
   before the gate; an ORT reference has been generated
   (logits `[1,900,256]`, pred_boxes `[1,900,4]`). The int64 `Clip`
   rows and the shape-inference dtype preservation landed 2026-10-19
   (see the P0 gate result above); the gate is unblocked and pending
   re-run.

## Environment notes

- **cuDNN gates.** Build with `CUDNN_ROOT=/tmp/opencode/cudnn` and run
  with
  `LD_LIBRARY_PATH=$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib:/tmp/opencode/cudnn/lib64`.
  A bare `ctest` in `build-cudnn` aborts with
  `Cannot load symbol cublasLtGetVersion` on this host; the CUDA-13 pip
  runtime supplies the matching `cublasLt`.
- **Zero-extent tensors.** RF-DETR's transformer concatenates a
  zero-length slice. `Concat` accepts zero-length operands on CPU, CUDA
  and OpenCL (a zero extent contributes nothing); a zero-only output
  returns without a launch.
- **OpenCL fp64.** `GlobalAveragePool` accumulates in double and
  requires `cl_khr_fp64`; a device without double-precision support
  fails with a named error before the kernel builds.
- **TF32.** CUDA math modes now truly forbid TF32 (`CUBLAS_PEDANTIC_MATH`,
  `CUDNN_FMA_MATH`). `NVIDIA_TF32_OVERRIDE=0` is no longer needed; before
  the fix, TF32 noise flipped RF-DETR's near-tie TopK order and changed
  the outputs by O(0.5).

## Sources

- Kernel tables: `src/cuda/table.cu`, `src/opencl/table.cpp`,
  `src/hip/table.cpp`, `src/kernels/cpu_kernels.cpp`.
- NERT model gates: `specs/2026-10-17-task-models/validation.md`,
  `specs/2026-10-17-n1g-yolo-gpu/validation.md`,
  `specs/2026-10-18-closure-cuda/validation.md`,
  `specs/2026-10-18-model-gates/validation.md`,
  `specs/2026-10-18-opset18-conversions/validation.md`,
  `specs/2026-10-18-opencl-flatten-gemm/validation.md`,
  `specs/2026-10-18-opencl-conv/validation.md`,
  `specs/2026-10-18-opencl-kernel-matrix/validation.md`.
- RAFT CPU parity: `specs/2026-10-10-raft-open-vocab/validation.md`.
- Consumer adapter: `neuriplo/backends/nert/src/NertInfer.cpp`,
  `neuriplo/docs/backends.yaml`.
- Inventory method: NERT loader probe over the local ONNX exports
  (post-fold node/op counts), plus `onnx.load(..., load_external_data=False)`
  for the opset/node metadata of exports NERT rejects. Raw inventory
  paths are recorded in the analysis run log; rerun after changing an
  export.
