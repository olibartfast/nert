# Coverage matrix — task models × devices

Date: 2026-10-18. Measured pass against the working tree. The CUDA
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
| OpenCL | 25 | `Add`, `Div`, `Mul`, `Relu`, `Reshape`, `Sigmoid`, `Softmax`, `Sub`, `Tanh`, `Transpose`, `Slice`, `Concat`, `Shape`, `Unsqueeze`, `Gather`, `MaxPool`, `AveragePool`, `GlobalAveragePool`, `Flatten`, `Gemm`, `Conv`, `MatMul`, `LayerNormalization`, `Expand`, `Where`. |
| HIP | 9 | `Add`, `Div`, `Mul`, `Relu`, `Reshape`, `Sigmoid`, `Softmax`, `Sub`, `Tanh`. Lags OpenCL by the movement six and the three pools. |

The CUDA table being op-complete is the key result: **there are no
missing CUDA op types to add for any current model.** New CUDA kernel
work cannot unlock these models; full-model gates and variant coverage
can. The portable layers are where kernel coverage is missing.

## Models NERT can load (opset 18)

| Model | Task | NERT nodes (ONNX) | CPU | CUDA | OpenCL | HIP | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| ResNet-18 | classification | 49 (49) | PASS | PASS | PASS | GAP(5) / 24 | CUDA gate V-1; OpenCL gate 2026-10-19: top-1 107 = ORT, logits maxdiff 3.695e-06. |
| YOLO26n | detection | 421 (485) | PASS | PASS | GAP(9) / 28 | GAP(21) / 197 | CUDA gate R-6; 6/6 confident rows. |
| OWLv2-base-patch16 | open-vocab detection | 952 (1219) | PASS | PASS | GAP(12) / 42 | GAP(24) / 453 | CPU and CUDA parity V-3/V-3b. |
| RF-DETR seg-small | detection + segmentation | 1255 (1785) | PASS | PASS | GAP(21) / 153 | GAP(33) / 701 | CUDA gate 2026-10-18: boxes 9.9e-04, labels 3.1e-04, top-1 100%, mask IoU 0.9995. |
| RAFT-large (torchvision) | optical flow | 1299 (1836) | PASS | PASS | GAP(9) / 119 | GAP(19) / 501 | Real opset-18 export 2026-10-19; CPU max 1.317e-04, CUDA max 1.344e-04 vs ORT (tol 5.324e-03). |

RAFT-large was re-provisioned as a real torchvision export (opset 18, 1,836 ONNX nodes, 1,299 after folding) on 2026-10-19. The earlier 156-node row came from `/tmp/opencode/ref.onnx`, a FoundationPose-style registration refiner (inputs `A`/`B` 6×160×160, outputs `trans`/`rot`), not RAFT; that misattribution is corrected here. Both CPU and CUDA full-model gates use the new export (1.317e-04 / 1.344e-04 vs ORT).

### Missing OpenCL rows by model

Counts are NERT post-fold node touches. HIP is a subset of this table
plus the OpenCL-only movement/pool rows; its exact gaps follow.

- **YOLO26n — 9 types, 28 nodes:** `Split=12`, `Cast=3`,
  `GatherElements=3`, `ConstantOfShape=2`, `Equal=2`, `Resize=2`,
  `TopK=2`, `Mod=1`, `ReduceMax=1`.
- **OWLv2 — 12 types, 42 nodes:** `IsNaN=12`, `Cast=8`,
  `ConstantOfShape=4`, `Equal=4`, `Erf=4`, `ReduceL2=3`,
  `And=2`, `ArgMax=1`, `Einsum=1`, `Elu=1`,
  `Greater=1`, `LessOrEqual=1`.
- **RF-DETR seg-small — 21 types, 153 nodes:** `Sqrt=48`,
  `Cast=38`, `Erf=17`, `ConstantOfShape=6`, `ReduceSum=5`, `Split=5`,
  `Cos=4`, `Equal=4`, `GridSample=4`, `Sin=4`, `Exp=3`,
  `Not=3`, `Tile=3`, `Greater=2`, `And=1`, `Einsum=1`,
  `GatherElements=1`, `Less=1`, `ReduceMax=1`, `Resize=1`, `TopK=1`.
- **RAFT-large — 9 types, 119 nodes:** `GridSample=48`, `Split=48`,
  `InstanceNormalization=15`, `Cast=2`, `Range=2`,
  `ConstantOfShape=1`, `Pad=1`, `ReduceSum=1`, `Tile=1`.

### Missing HIP rows by model

HIP carries only the original nine rows, so every OpenCL gap applies
and the movement six plus the pools are also missing. The extra
HIP-only gap beyond the OpenCL list is:

- **ResNet-18 — 5 types, 24 nodes:** `Conv=20`,
  `GlobalAveragePool=1`, `MaxPool=1`, `Flatten=1`, `Gemm=1`.
- **YOLO26n — 21 types, 197 nodes:** OpenCL gap plus `Conv=102`,
  `Concat=28`, `Unsqueeze=11`, `Transpose=6`, `MatMul=4`, `MaxPool=3`,
  `Slice=2`, `Expand=8`, `Where=2`, `Gather=1`, `Shape=1`, `Flatten=1`.
- **OWLv2 — 24 types, 453 nodes:** OpenCL gap plus `Conv=1`,
  `MatMul=202`, `Transpose=97`, `LayerNormalization=53`, `Where=28`,
  `Gather=11`, `Concat=5`, `Shape=5`, `Slice=3`, `Expand=3`,
  `Unsqueeze=1`, `Flatten=2`.
- **RF-DETR seg-small — 33 types, 701 nodes:** OpenCL gap plus
  `Conv=14`, `MatMul=158`, `Transpose=133`, `Slice=72`,
  `LayerNormalization=56`, `Concat=38`, `Shape=32`, `Unsqueeze=23`,
  `Expand=8`, `Where=6`, `Gather=4`, `Gemm=4`.
- **RAFT-large — 19 types, 501 nodes:** OpenCL gap plus `Conv=190`,
  `Concat=135`, `Transpose=27`, `Gather=7`, `Unsqueeze=7`, `Shape=5`,
  `Slice=4`, `AveragePool=3`, `Expand=3`, `MatMul=1`.

## Converted exports now loadable (opset-18 conversion, 2026-10-18)

`onnx.version_converter.convert_version(model, 18)` plus
`onnx.checker`; each row then loaded through NERT's own loader and
shape inference. **CPU and CUDA op coverage is complete for every row**
(0 missing types), but no reference gate has been run for these
variants — they are OPS, not PASS. OpenCL/HIP gaps are ONNX op-type
counts (Constant excluded).

| Model | Source opset | NERT nodes | OpenCL gap | HIP gap |
| --- | ---: | ---: | --- | --- |
| Grounding DINO | 17 | 19,224 | 33 / 4,072 | 45 / 15,336 |
| RF-DETR keypoint-preview | 17 | 2,355 | 21 / 370 | 33 / 1,486 |
| RF-DETR seg-nano-576 | 17 | 1,216 | 21 / 154 | 33 / 675 |
| RF-DETR nano-1101 | 17 | 1,002 | 19 / 133 | 31 / 567 |
| FoundationPose scorer | 17 | 145 | 2 / 2 | 10 / 98 |
| YOLO26s | 20 | 391 | 8 / 15 | 17 / 176 |
| YOLO26x | 19 | 606 | 8 / 24 | 17 / 247 |
| YOLO26n-depth | 20 | 328 | 6 / 21 | 11 / 139 |
| YOLOv8n-pose | 20 | 267 | 2 / 10 | 7 / 110 |
| YOLO11s | 12 | 321 | 2 / 12 | 8 / 132 |
| YOLO26s-pose | 12 | 401 | 2 / 13 | 8 / 164 |

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
| YOLO26n | PASS | PASS | GAP(9/28) | GAP(21/197) | R-6 (6/6 confident rows, min IoU 0.9998039; detection-identical). |
| OWLv2 | PASS | PASS | GAP(12/42) | GAP(24/453) | V-3/V-3b (logits exact; boxes 1.9e-04; 0 rows over tolerance). |
| RF-DETR seg-small | PASS | PASS | GAP(21/153) | GAP(33/701) | CUDA gate boxes 9.938e-04, labels 3.080e-04, top-1 100%, mask IoU 0.999528. |
| RAFT-large | PASS | PASS | GAP(9/119) | GAP(19/501) | Full gate 2026-10-19 on a real `bus.jpg` pair: CPU maxdiff 1.317e-04 (1,074 s), CUDA 1.344e-04; tolerance 5.324e-03. |
| Grounding DINO | OPS | OPS | GAP(35/5,235) | GAP(45/15,336) | Loads (19,224 folded nodes), CPU/CUDA op-complete; reference gate pending. |
| Converted YOLO/RF-DETR variants | OPS | OPS | per converted table | per converted table | Load + shape inference only; no reference gate yet. |
| DETR/ViT/ViTPose/Depth-Anything/VideoMAE/ViViT/TimeSformer | BLOCKED | BLOCKED | BLOCKED | BLOCKED | No local export (searched 2026-10-18). |
| DETR family | Recorded (roadmap Phase 2) | Not run | Opset-18 export/harness still the gate. |

## Portable-backend priority (OpenCL)

Shared by the four primary loadable models above, sorted by model count
then node touches. This ranks OpenCL; HIP additionally lacks the movement
six and the pools, so its unlock sets are larger. The converted variants
have their own gap columns in the table above.

| Op type | Models using it | Node touches | Why it matters |
| --- | ---: | ---: | --- |
| `Cast` | 3 | 49 | Dtype crosses. |
| `ConstantOfShape` | 3 | 12 | Shape-generated constants. |
| `Equal` | 3 | 10 | Masks. |
| `Erf` | 2 | 21 | Transformer activations. |
| `Split` | 2 | 17 | Multi-head reshapes. |
| `GatherElements` | 2 | 4 | Index gathers. |
| `And` | 2 | 3 | Masks. |

Remaining op types are used by one or two models:
`Greater` (2/3), `Resize` (2/3), `TopK` (2/3), `Einsum` (2/2),
`ReduceMax` (2/2), `Sqrt` (1/48), `IsNaN` (1/12), `ReduceSum` (1/5),
`Cos` (1/4), `GridSample` (1/4), `Sin` (1/4), `Exp` (1/3), `Not` (1/3),
`ReduceL2` (1/3), `Tile` (1/3),
`ArgMax` (1/1), `Elu` (1/1), `Less` (1/1),
`LessOrEqual` (1/1), `Mod` (1/1).

Model-centric milestones:

| Model unlock | OpenCL | HIP |
| --- | ---: | ---: |
| ResNet-18 | 0 | 5 |
| RAFT-large | 9 | 19 |
| YOLO26n | 9 | 21 |
| OWLv2 | 12 | 24 |
| RF-DETR seg-small | 21 | 33 |

ResNet-18 is op-complete and gates on OpenCL (0 missing types). Expand
and Where landed 2026-10-19, taking YOLO26n to 9 types, OWLv2 to 12 and
RF-DETR to 21. The next shared rows are `Cast`, `ConstantOfShape`,
`Equal`, `Erf` and `Split`.

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
3. **Pending gates.** RF-DETR's CUDA gate now passes (2026-10-18).
   RAFT is blocked on a missing export, not on kernels.
   OWLv2/YOLO26n/ResNet gates pass.
4. **Grounding DINO.** Converted and loading (19,224 folded nodes; all
   CPU/CUDA op types covered). CPU parity and the CUDA gate still need
   reference inputs; it is the largest open-vocab export.

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
