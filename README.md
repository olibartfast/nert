# NERT (Neuriplo Engine Runtime)

NERT is a first-party ONNX inference runtime: an interpreter for ONNX
opset 18. It computes in float32, with int64/bool shape data and I/O, and
is standard-library-only by default (tests use GoogleTest; opt-in device
layers add their own toolkit dependency, linked PRIVATE).

It is an interpreter: each model load parses the ONNX graph, infers tensor
shapes for the requested input, plans a single arena allocation, then executes
the nodes one by one in topological order. It is not a plan-building compiler;
there is no graph lowering, fusion (beyond load-time constant folding), or
code generation.

The device is chosen once per loaded graph. A graph that cannot run entirely
on the requested device is rejected when it is loaded; there is no per-operator
fallback to another device at run time.

The CPU kernels are the executable correctness oracle for the device
layers. They are deliberately unoptimized and are not a performance target;
they exist so device layers can be checked for numerical agreement rather
than re-derive operator semantics. Each device layer runs the same model
with the same plan and must match the CPU oracle within tolerance.

## Status

Version 0.2.0 (unreleased, breaking): namespace `engine` → `nert`, public
headers `include/engine/` → `include/nert/`, all C++ filenames lowercase
`snake_case`, CTest suites `engine_*` → `nert_*`. No behavior change; see
[CHANGELOG](CHANGELOG.md).

CPU coverage has been validated against ONNX Runtime through
[Neuriplo](https://github.com/olibartfast/neuriplo) on the YOLO families,
ViT, ViTPose, Depth Anything V2, VideoMAE, ViViT, TimeSformer, the DETR
family (RT-DETR v4, D-FINE/DEIM, DEIMv2, RF-DETR, RF-DETR-Seg), RAFT
(`InstanceNormalization`, Tier 4), and the Tier-5 open-vocabulary kernel
set. Open-vocabulary end-to-end parity is the open integration gate.

## Documentation

- [Memory footprint profiling](docs/memory-footprint-profiling.md) —
  planned model fit/sizing report (API + CLI), the accounting model, and
  a current plan-level workaround.

## Devices

`CpuDevice()` always exists. `CudaDevice()`, `OpenCLDevice()`, and
`HipDevice()` exist only when the matching build flag is on, and device
kernel coverage is partial: each layer implements a subset of the CPU
table, checked against the CPU oracle within tolerance. Pick devices via
`nert::CpuDevice()` / `nert::CudaDevice()` / `nert::OpenCLDevice()` /
`nert::HipDevice()` in `include/nert/device.hpp`.

## Layout

- `include/nert/`: public headers (namespace `nert`): graph IR, model
  loader, shape inference, memory plan, constant folding, device seam, and
  executor.
- `src/`: loader, shape inference, memory planner, constant folding,
  executor, and the `cpu_device` / `cuda_device` / `opencl_device` /
  `hip_device` layers.
- `src/kernels/`: the CPU operator kernels behind the CPU kernel table.
- `src/cuda/`, `src/opencl/`, `src/hip/`: opt-in device kernels (built
  only with the matching flag).
- `test/`: unit tests for the loader, shapes, plan, device, kernels,
  constant folding, and executor, plus `cuda_*` / `opencl_*` / `hip_*`
  device tests built only with the matching flag.
- `bench/`: opt-in shipped-kernel benchmark driver; see
  `bench/README.md`.
- `tools/provision_resnet18.py`: exports the ResNet-18 test fixture.
- `scripts/format.sh`: clang-format check.
- `docs/`: user guides (memory footprint profiling is specified but not
  yet implemented).
- `specs/`: mission, tech stack, roadmap, and per-feature packets.

## Build

```bash
cmake -S . -B build
cmake --build build
```

The library target is `nert` (alias `nert::nert`). Tests are built when NERT is
the top-level project; control them with `-DNERT_BUILD_TESTS=ON|OFF`
(`BUILD_INFERENCE_ENGINE_TESTS` is still honoured when a parent project sets it).

Opt-in flags (all `OFF` by default, so the default build stays
standard-library-only):

| Flag | Dependency | Effect |
| --- | --- | --- |
| `NERT_ENABLE_CUDA=ON` | CUDA toolkit (+ optional cuDNN for `Conv`) | Builds `src/cuda_device.cpp` + `src/cuda/` and the `nert_cuda_*` tests |
| `NERT_ENABLE_OPENCL=ON` | OpenCL headers + ICD loader | Builds `src/opencl_device.cpp` + `src/opencl/` and the `nert_opencl_*` tests |
| `NERT_ENABLE_HIP=ON` | CUDA compiler (NVIDIA target) + HIP headers | Builds `src/hip_device.cpp` + `src/hip/` and the `nert_hip_*` tests |
| `NERT_BUILD_BENCH=ON` | None beyond the enabled devices | Builds the `bench/runtime` driver (`nert_bench_runtime`) |

```bash
cmake -S . -B build-cuda -DNERT_ENABLE_CUDA=ON -DNERT_BUILD_BENCH=ON
cmake --build build-cuda --target nert_bench_runtime
```

## Test

The CPU tests read an opset-18 ResNet-18 model from `NERT_FIXTURE` (environment
variable or CMake cache variable). Provision it with torch and torchvision
0.27.0 installed, then run ctest:

```bash
python3 tools/provision_resnet18.py fixtures
NERT_FIXTURE=fixtures/resnet18.onnx ctest --test-dir build --output-on-failure
```

Device tests (`nert_cuda_*`, `nert_opencl_*`, `nert_hip_*`) are built only
with the matching flag and SKIP at run time without a usable GPU, so
toolkit-only runners stay green. The bench driver runs as
`ctest -R bench_runtime` (missing GPU prints `SKIP:` and exits 0).

Check formatting with `scripts/format.sh --check` (or `--fix`).

## Opset policy

NERT loads opset 18, ONNX domain only. Models exported at an older opset
are rejected at load — convert first, then load:

```python
import onnx
from onnx import version_converter
model = onnx.load("model_op16.onnx")
model = version_converter.convert_version(model, 18)
onnx.save(model, "model_op18.onnx")
```

## Install and use

```bash
cmake --install build --prefix /path/to/prefix
```

This installs the headers, the static library, and the CMake package files. A
consumer project:

```cmake
find_package(nert 0.2 REQUIRED)
target_link_libraries(app PRIVATE nert::nert)
```

```cpp
#include "nert/model_loader.hpp"
#include "nert/executor.hpp"
#include "nert/device.hpp"

nert::Graph graph = nert::LoadGraphFromFile("model_op18.onnx");
nert::Model model(graph, {{"input", {1, 3, 224, 224}}}, nert::CpuDevice());
```

The old target name `neuriplo_engine` remains available as a compatibility
alias of `nert`.

## Use from Neuriplo

[Neuriplo](https://github.com/olibartfast/neuriplo) consumes NERT as its
`NERT` inference backend, adapting it to Neuriplo's `InferenceInterface`. NERT
itself has no knowledge of Neuriplo.

## License

MIT; see [LICENSE](LICENSE).
