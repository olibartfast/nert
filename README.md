# NERT (Neuriplo Engine Runtime)

NERT is a first-party ONNX inference runtime: an interpreter for ONNX
opsets 11 and above. Each node resolves its effective schema, and the
version-dependent operator forms are handled in place. Newer declarations are
accepted only when every node resolves to a schema version the current kernels
can execute. It computes in float32, with
int64/bool shape data and I/O, and is standard-library-only by default (tests
use GoogleTest; opt-in device layers add their own toolkit dependency, linked
PRIVATE).

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

## Design reference

The typed-graph direction, static shape inference, constant folding, and
operator-level semantics are informed by OpenCV 5's new DNN engine:
[OpenCV 5 — A Brand-New DNN Engine](https://opencv.org/opencv-5/#h-the-headline-a-brand-new-dnn-engine).
NERT does not depend on OpenCV and does not copy its automatic engine
fallback or ONNX Runtime escape hatch; unsupported semantics fail explicitly.

## Status

Version 0.2.0 is unreleased; the breaking namespace, header, filename,
and CTest-suite renames are complete. See [CHANGELOG](CHANGELOG.md).

CPU and CUDA tables are complete at 72 op rows each. OpenCL has 68
rows; the four absent types (`Abs`, `Floor`, `GreaterOrEqual`, `Neg`)
are unused by the catalog entries that currently load. HIP is
unsupported for 0.2.0: nine rows, and the reference host has no ROCm
toolchain or AMD device.

Full-model gates, not op-table presence, define coverage. As of 2026-10-20,
22 of the 30 in-scope ONNX catalog entries pass on CPU, CUDA, and OpenCL; one
entry remains open (Grounding DINO: CPU/CUDA task-equivalence pass, OpenCL
backend gap), and 7 entries need an ONNX export.
Image understanding (GGUF/llama.cpp) and Gaussian
Splatting (LGM/GRM) are out of scope for this ONNX runtime. The catalog
is incomplete and 0.2.0 is not release-ready; see the
[coverage matrix](specs/2026-10-17-task-models/coverage.md).

## Documentation

- [Model and device coverage](specs/2026-10-17-task-models/coverage.md)
  — full-model gates, kernel-table baseline, and release scoreboard.
- [Catalog program](specs/2026-10-19-catalog-program/inventory.md) —
  32-entry Neuriplo task catalog, export feasibility, and open work.
- [Memory footprint profiling](docs/memory-footprint-profiling.md) —
  planned model fit/sizing report (API + CLI), the accounting model, and
  a current plan-level workaround.

## Devices

`CpuDevice()` always exists. `CudaDevice()`, `OpenCLDevice()`, and
`HipDevice()` exist only when the matching build flag is on. CPU and
CUDA implement the full 72-row table; CUDA `Conv` and `ConvTranspose`
require cuDNN. OpenCL implements 68 rows as the portable backend. HIP is
unsupported for 0.2.0 and remains a nine-row scaffold.

A device can load only graphs whose op types all have a row. Op-table
presence is a coverage statement, not a correctness claim; full-model
gate status is recorded in the coverage matrix. Pick devices via
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

The CPU tests read a provisioned ResNet-18 model from `NERT_FIXTURE`
(environment variable or CMake cache variable). The provided provisioning tool
exports the opset-18 model; a native opset-19 export with the same graph is
also accepted. Provision it with torch and torchvision 0.27.0 installed, then
run ctest:

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

NERT loads ONNX-domain opsets 11 and above. Each node carries its effective
schema, and the supported version-dependent forms stay in the graph. There is
no global load-time canonicalization pass:

- `ReduceMean`, `ReduceL2`, `ReduceMax`: pre-18 `axes` attribute is used in
  place (shape inference and the CPU/CUDA/OpenCL rows select the form from
  `Node::schema_version`); schema 18+ uses the optional int64 input.
- `ReduceSum`, `Squeeze`, `Unsqueeze`: pre-13 `axes` attribute is used in
  place; schema 13+ uses the input form.
- `Split`: pre-13 `split` attribute is used in place; schema 13+ uses the
  input form.
- `Softmax`: pre-13 2-D coercion and the old default axis of 1 are
  implemented directly in the kernel; schema 13+ uses single-axis semantics.
- `ScatterND`: `reduction="none"` is accepted in place; other reductions are
  rejected because the kernel implements only the overwrite form.

Models below opset 11 are rejected at load; convert them first:

```python
import onnx
from onnx import version_converter
model = onnx.load("model_op10.onnx")
model = version_converter.convert_version(model, 18)
onnx.save(model, "model_op18.onnx")
```

Legacy `Softmax` (opset < 13) needs the input rank from the model's shape
metadata (`value_info` for intermediates) so the loader can validate `axis`;
a model that omits that information is rejected with a message asking for
`value_info` or a conversion.

There is no upper opset ceiling. For a declaration above 18, each node
resolves its **effective schema version** (the latest ONNX schema at or
before the declared opset) from a committed compatibility table. The node is
accepted when that schema is equivalent to the opset-18 implementation, or
when it only adds attributes that the node does not use. A newer attribute
or value on an implemented operator fails at load with the node, operator,
declared opset, effective version, and the missing capability. Operators and
dtypes outside the implemented set keep their existing explicit rejections.
Current guards include
`Cast.saturate` (19+), `Cast.round_mode` (24+), `Pad.mode="wrap"` (19+),
`AveragePool.dilations` (19+), `Range.stash_type` (27+), and nonzero
`ReduceMax.noop_with_empty_axes` at schema 18. `GridSample` at
effective 20+ is rejected because its mode default and value domain changed;
the opset-16 behavior is implemented. Nothing is silently ignored.

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
