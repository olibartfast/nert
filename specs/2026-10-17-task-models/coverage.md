# Coverage matrix — task models × devices

Date: 2026-10-18. First measured pass against the working tree, including
the uncommitted CUDA-closure slice; OpenCL movement rows
(Transpose/Slice/Concat/Shape/Unsqueeze/Gather) folded in same-day.
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
| CUDA | 72 | Identical set to CPU. `Conv` and `ConvTranspose` are cuDNN-gated; a CUDA build without cuDNN lacks those two rows. |
| OpenCL | 18 | `Add`, `Div`, `Mul`, `Relu`, `Reshape`, `Sigmoid`, `Softmax`, `Sub`, `Tanh`, `Transpose`, `Slice`, `Concat`, `Shape`, `Unsqueeze`, `Gather`, `MaxPool`, `AveragePool`, `GlobalAveragePool`. |
| HIP | 9 | Same set as OpenCL. |

The CUDA table being op-complete is the key result: **there are no
missing CUDA op types to add for any current model.** New CUDA kernel
work cannot unlock these models; full-model gates and variant coverage
can. The portable layers are where kernel coverage is missing.

## Models NERT can load (opset 18)

| Model | Task | NERT nodes (ONNX) | CPU | CUDA | OpenCL / HIP | Notes |
| --- | --- | ---: | --- | --- | --- | --- |
| ResNet-18 | classification | 49 (49) | PASS | PASS | GAP(3) / 22 | CUDA gate V-1; CI fixture. |
| YOLO26n | detection | 421 (485) | PASS | PASS | GAP(14) / 145 | CUDA gate R-6; 6/6 confident rows. |
| OWLv2-base-patch16 | open-vocab detection | 952 (1219) | PASS | PASS | GAP(18) / 331 | CPU and CUDA parity V-3/V-3b. |
| RF-DETR seg-small | detection + segmentation | 1255 (1785) | PASS | OPS | GAP(27) / 399 | CUDA op coverage complete; full gate pending. |
| RAFT (ref component) | optical flow | 156 (156) | PASS | OPS | GAP(6) / 38 | Full RAFT CPU parity pass; CUDA gate pending after InstanceNormalization landed. |

### Missing OpenCL/HIP rows by model

Counts are NERT post-fold node touches. The same list applies to OpenCL
and HIP because their tables currently carry the same nine op types.

- **ResNet-18 — 3 types, 22 nodes:** `Conv=20`, `Flatten=1`,
  `Gemm=1`.
- **YOLO26n — 14 types, 145 nodes:** `Conv=102`, `Split=12`,
  `Expand=8`, `MatMul=4`, `Cast=3`, `GatherElements=3`,
  `ConstantOfShape=2`, `Equal=2`, `Resize=2`, `TopK=2`, `Where=2`,
  `Flatten=1`, `Mod=1`, `ReduceMax=1`.
- **OWLv2 — 18 types, 331 nodes:** `MatMul=202`,
  `LayerNormalization=53`, `Where=28`, `IsNaN=12`, `Cast=8`,
  `ConstantOfShape=4`, `Equal=4`, `Erf=4`, `Expand=3`, `ReduceL2=3`,
  `And=2`, `Flatten=2`, `ArgMax=1`, `Conv=1`, `Einsum=1`, `Elu=1`,
  `Greater=1`, `LessOrEqual=1`.
- **RF-DETR seg-small — 27 types, 399 nodes:** `MatMul=158`,
  `LayerNormalization=56`, `Sqrt=48`, `Cast=38`, `Erf=17`, `Conv=14`,
  `Expand=8`, `ConstantOfShape=6`, `Where=6`, `ReduceSum=5`, `Split=5`,
  `Cos=4`, `Equal=4`, `Gemm=4`, `GridSample=4`, `Sin=4`, `Exp=3`,
  `Not=3`, `Tile=3`, `Greater=2`, `And=1`, `Einsum=1`,
  `GatherElements=1`, `Less=1`, `ReduceMax=1`, `Resize=1`, `TopK=1`.
- **RAFT ref — 6 types, 38 nodes:** `Conv=15`, `MatMul=12`,
  `LayerNormalization=4`, `Squeeze=3`, `Gemm=2`, `ReduceMean=2`.

## Exports NERT cannot load (opset != 18)

NERT requires ONNX-domain opset 18. 19 of 24 inventoried export files are
not loadable today; their op-type coverage against the CPU table is
complete once `Constant` is folded, so the blocker is the export/opset,
not a missing CPU kernel.

| Export/file | Task | Opset | ONNX nodes | Blocker |
| --- | --- | ---: | ---: | --- |
| Grounding DINO | open-vocab detection | 17 | 31,821 | Convert to 18; CPU/CUDA gate pending; largest export. |
| RF-DETR seg-small (deep-infer) | detection + segmentation | 17 | 1,879 | Convert to 18; duplicate of the NERT-side export. |
| RF-DETR seg-nano-576 | detection + segmentation | 17 | 1,710 | Convert to 18. |
| RF-DETR nano-1101 | detection | 17 | 1,434 | Convert to 18. |
| RF-DETR keypoint-preview | keypoint | 17 | 1,392 / 3,542 | Two exports; convert to 18. |
| RAFT components (`ref2`, `refiner`, `rw`, `scorer`, `sw`) | optical flow | 17 | 363 / 382 / 377 / 209 / 300 | Convert to 18; `ref.onnx` already loads. |
| YOLO26s | detection | 20 | 391 | Convert to 18. |
| YOLO26n-depth | depth | 20 | 348 | Convert to 18. |
| YOLOv8n-pose | pose | 20 | 267 | Convert to 18. |
| YOLO11s | detection | 12 | 321 | Convert to 18. |
| YOLO26s-pose | pose | 12 | 401 / 418 | Two exports; convert to 18. |
| YOLO26x | detection | 19 | 606 | Convert to 18. |

The DETR family (RT-DETR/D-FINE/DEIM/DEIMv2), ViT, ViTPose,
Depth-Anything-V2, VideoMAE, ViViT, TimeSformer, and generic
segmentation/pose families have CPU coverage recorded in the roadmap but
no NERT-loadable opset-18 export in this tree, so they get no row here
until an export exists. See the pending V-2/V-3 entries in
[`validation.md`](validation.md).

## Full-model gates

| Model | CPU parity | CUDA full gate | Evidence / blocker |
| --- | --- | --- | --- |
| ResNet-18 | PASS | PASS | V-1 (top-1 identical; max diff 2.167e-03, classification-identical). |
| YOLO26n | PASS | PASS | R-6 (6/6 confident rows, min IoU 0.9998039; detection-identical). |
| OWLv2 | PASS | PASS | V-3/V-3b (logits exact; boxes 1.9e-04; 0 rows over tolerance). |
| RF-DETR seg-small | PASS | OPS | Kernels complete; full CUDA gate not recorded. |
| RAFT | PASS | OPS | InstanceNormalization row landed; full CUDA gate not recorded. |
| Grounding DINO | Pending | Pending | Opset-18 conversion and reference run needed. |
| Other YOLO families | Recorded (model zoo) | Not run | Local exports are opset 12/19/20; conversion first. |
| DETR family | Recorded (roadmap Phase 2) | Not run | Opset-18 export/harness still the gate. |

## Portable-backend priority (OpenCL/HIP)

Shared by the five loadable models, sorted by model count then node
touches:

| Op type | Models using it | Node touches | Why it matters |
| --- | ---: | ---: | --- |
| `Conv` | 5 | 152 | Blocks every loadable model; hardest portable kernel. |
| `MatMul` | 4 | 376 | Transformer detection, OWLv2, RAFT. |
| `LayerNormalization` | 3 | 113 | Both transformer detectors and RAFT. |
| `Cast` | 3 | 49 | Dtype crosses. |
| `Where` | 3 | 36 | Masks. |
| `Expand` | 3 | 19 | Broadcast. |
| `ConstantOfShape` | 3 | 12 | Shape-generated constants. |
| `Equal` | 3 | 10 | Masks. |
| `Gemm` | 3 | 7 | ResNet classifier, RF-DETR, RAFT. |
| `Flatten` | 3 | 4 | ResNet/YOLO/OWLv2. |

Remaining op types are used by one or two models: `Erf` (2/21),
`Split` (2/17), `GatherElements` (2/4), `And` (2/3),
`Greater` (2/3), `Resize` (2/3), `TopK` (2/3), `Einsum` (2/2),
`ReduceMax` (2/2), `Sqrt` (1/48), `IsNaN` (1/12), `ReduceSum` (1/5),
`Cos` (1/4), `GridSample` (1/4), `Sin` (1/4), `Exp` (1/3), `Not` (1/3),
`ReduceL2` (1/3), `Squeeze` (1/3), `Tile` (1/3), `ReduceMean` (1/2),
`ArgMax` (1/1), `Elu` (1/1), `Less` (1/1),
`LessOrEqual` (1/1), `Mod` (1/1).

Model-centric milestones:

| Model unlock | Op types still needed |
| --- | ---: |
| ResNet-18 | 3 |
| RAFT ref | 6 |
| YOLO26n | 14 |
| OWLv2 | 18 |
| RF-DETR seg-small | 27 |

`Conv` is required by all five, so no full model can run on OpenCL/HIP
until it lands. The useful order is therefore: Flatten/Gemm rows, then
`Conv`, then the transformer-specific rows (the shared movement six
and pools landed 2026-10-18).

## Non-kernel blockers

1. **Opset policy.** NERT loads opset 18 only. 19 of 24 inventoried
   exports need `onnx.version_converter.convert_version(model, 18)` and a
   re-export/reference run before any kernel work can be claimed for
   those families.
2. **Consumer adapter.** Neuriplo's NERT backend is CPU-only:
   `backends/nert/src/NertInfer.cpp:81-82` throws `"NERT runs on CPU only
   in this phase"` when `use_gpu` is true, and
   `neuriplo/docs/backends.yaml` marks NERT `gpu: false`. No Neuriplo
   task family can dispatch to a NERT GPU device until that adapter
   selects `CudaDevice`/`OpenCLDevice`/`HipDevice` and the GPU CI leg
   exists. NERT kernel coverage alone does not change the consumer
   matrix.
3. **Pending gates.** RF-DETR and RAFT have complete CUDA op coverage but
   no full-model CUDA gate. OWLv2/YOLO26n/ResNet gates pass.
4. **Grounding DINO.** Opset 17, 31,821 nodes, CPU parity and CUDA gate
   both pending; it is the largest open-vocab export.

## Sources

- Kernel tables: `src/cuda/table.cu`, `src/opencl/table.cpp`,
  `src/hip/table.cpp`, `src/kernels/cpu_kernels.cpp`.
- NERT model gates: `specs/2026-10-17-task-models/validation.md`,
  `specs/2026-10-17-n1g-yolo-gpu/validation.md`,
  `specs/2026-10-18-closure-cuda/validation.md`.
- RAFT CPU parity: `specs/2026-10-10-raft-open-vocab/validation.md`.
- Consumer adapter: `neuriplo/backends/nert/src/NertInfer.cpp`,
  `neuriplo/docs/backends.yaml`.
- Inventory method: NERT loader probe over the local ONNX exports
  (post-fold node/op counts), plus `onnx.load(..., load_external_data=False)`
  for the opset/node metadata of exports NERT rejects. Raw inventory
  paths are recorded in the analysis run log; rerun after changing an
  export.
