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
