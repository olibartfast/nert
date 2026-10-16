# NERT (Neuriplo Engine Runtime)

NERT is a first-party ONNX inference runtime: a CPU reference interpreter for
ONNX opset 18. It computes in float32, with int64/bool shape data and I/O, and
depends on nothing beyond the C++ standard library (the tests use GoogleTest).

It is an interpreter: each model load parses the ONNX graph, infers tensor
shapes for the requested input, plans a single arena allocation, then executes
the nodes one by one in topological order. It is not a plan-building compiler;
there is no graph lowering, fusion, or code generation.

The device is chosen once per loaded graph. A graph that cannot run entirely
on the requested device is rejected when it is loaded; there is no per-operator
fallback to another device at run time.

The CPU kernels are the executable correctness oracle for a future accelerated
path. They are deliberately unoptimized and are not a performance target; they
exist so later device layers can be checked for numerical agreement rather than
re-derive operator semantics.

## Status

Version 0.1.0 (unreleased). NERT has been validated against ONNX Runtime
through [Neuriplo](https://github.com/olibartfast/neuriplo) on the YOLO
families, ViT, ViTPose, Depth Anything V2, VideoMAE, ViViT, TimeSformer, and
the DETR family: RT-DETR v4, D-FINE/DEIM, DEIMv2, RF-DETR, and RF-DETR-Seg.
RAFT and open-vocabulary models are in progress.

## Layout

- `include/engine/`: public headers (namespace `engine`): graph IR, model
  loader, shape inference, memory plan, device seam, and executor.
- `src/`: implementation of the loader, shape inference, memory planner,
  device, and executor.
- `src/kernels/`: the CPU operator kernels behind the device kernel table.
- `test/`: unit tests for the loader, shapes, plan, device, kernels, constant
  folding, and executor.
- `tools/provision_resnet18.py`: exports the ResNet-18 test fixture.
- `scripts/format.sh`: clang-format check.

## Build

```bash
cmake -S . -B build
cmake --build build
```

The library target is `nert` (alias `nert::nert`). Tests are built when NERT is
the top-level project; control them with `-DNERT_BUILD_TESTS=ON|OFF`
(`BUILD_INFERENCE_ENGINE_TESTS` is still honoured when a parent project sets it).

## Test

The tests read an opset-18 ResNet-18 model from `NERT_FIXTURE` (environment
variable or CMake cache variable). Provision it with torch and torchvision
0.27.0 installed, then run ctest:

```bash
python3 tools/provision_resnet18.py fixtures
NERT_FIXTURE=fixtures/resnet18.onnx ctest --test-dir build --output-on-failure
```

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
find_package(nert 0.1 REQUIRED)
target_link_libraries(app PRIVATE nert::nert)
```

```cpp
#include "engine/Engine.hpp"
```

For now, the old target name `neuriplo_engine` remains available as an alias of
`nert` when NERT is built as a subproject.

## Use from Neuriplo

[Neuriplo](https://github.com/olibartfast/neuriplo) consumes NERT as its
`NERT` inference backend, adapting it to Neuriplo's `InferenceInterface`. NERT
itself has no knowledge of Neuriplo.

## Licence

MIT; see [LICENSE](LICENSE).
