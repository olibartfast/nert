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
  `NERT_HAS_CUDNN` only; non-zero output_padding and asymmetric pads
  rejected on CUDA). Every CPU op now has a CUDA row (`ConvTranspose`
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
- RAFT-large: real torchvision opset-18 export provisioned (1,836 ONNX
  nodes) and the full CPU+CUDA gate passes against ORT — flow maxdiff
  1.317e-04 (CPU) / 1.344e-04 (CUDA), tolerance 5.324e-03, on a real
  image pair. The earlier
  `ref.onnx` was a FoundationPose refiner, not RAFT.
- OpenCL full-model execution: handle-based devices now back each planned
  arena offset with its own buffer (`cl_mem` handles cannot be offset);
  two-output regression test. **ResNet-18 is the first full model to pass
  on OpenCL** — top-1 107 = ORT, logits maxdiff 3.695e-06 in 54 ms.

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
