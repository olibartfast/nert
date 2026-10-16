# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added
- First-party ONNX inference runtime (NERT, the Neuriplo Engine Runtime):
  model loader, shape inference, single-arena memory planner, constant
  folding, device seam, and executor, targeting ONNX opset 18.
- CPU reference kernels for the operator set needed by the YOLO families, ViT,
  ViTPose, Depth Anything V2, VideoMAE, ViViT, TimeSformer, and the DETR
  family (RT-DETR v4, D-FINE/DEIM, DEIMv2, RF-DETR, RF-DETR-Seg).
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
