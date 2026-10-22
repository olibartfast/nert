# NERT Technical Stack

> Status: constitution. Derived on 2026-10-10 from Neuriplo's `specs/tech-stack.md`
> (First-Party NERT section) and the native-engine packets. If this document
> and the implementation disagree, the implementation and its executable
> checks win.

## Stack at a Glance

| Area | Choice | Practical consequence |
| --- | --- | --- |
| Library | C++17 static library `nert`, alias `nert::nert`, compatibility alias `neuriplo_engine` | Do not raise the language standard without an explicit decision. Public headers live under `include/nert/` in namespace `nert`. C++ filenames use lowercase `snake_case`. |
| Build | CMake 3.16 or newer | A top-level build enables tests by default through `NERT_BUILD_TESTS`; a consumed build disables them unless requested. `BUILD_INFERENCE_ENGINE_TESTS` is still accepted for Neuriplo compatibility. |
| Runtime dependencies | C++ standard library only | No new runtime dependency without explicit approval. GoogleTest is test-only; PyTorch and torchvision 0.27.0 are fixture tooling and are never linked into the runtime. |
| ONNX input | Opsets 11 and above, ONNX domain only; every node checked against a full-history implementation catalog; hand-written protobuf wire reader | The loader resolves each node's effective schema and keeps the source form; shape inference and kernels select the version-aware behavior. The catalog comes from the reviewed manifest in `tools/opset_manifest.py` and rejects unimplemented attributes, values, and behaviors. There is no canonical opset. Models below 11 are converted with `onnx.version_converter`, as recorded in decision `Q-1`. |
| Compute | Float32 compute; int64 and bool for shape data and graph inputs/outputs | Reductions accumulate in double and round once per output. DOUBLE constants and value-info narrow to float32; BOOL constants and initializers load as bool. |
| Tests | GoogleTest with CTest; suites named `nert_*` | The standard fixture is an opset-18 ResNet-18 model provisioned by `tools/provision_resnet18.py` and selected with `NERT_FIXTURE`. |
| Quality | clang-format via `scripts/format.sh --check` | Keep the checked-in script's scope. CI pins clang-format 18. |

## Architecture

NERT is a sequential interpreter. A model moves through four stages:

1. `src/model_loader.cpp` parses the ONNX graph.
2. `src/shape_inference.cpp` resolves concrete shapes for the requested inputs.
3. `src/memory_planner.cpp` assigns intermediate tensors to one arena.
4. `src/executor.cpp` dispatches nodes in topological order through the
   selected device's kernel table.

The intermediate representation lives in `include/nert/graph.hpp`. It stores
nodes in execution order, a typed table of tensor values (float32, int64, and
bool), initializer storage with dtype tags, and the distinction between graph
inputs, initializers, and intermediates.

The device seam is defined in `include/nert/device.hpp`. The core is
device-agnostic; each device supplies an allocator, host/device transfers, and
an operator-keyed kernel table. The CPU implementation lives in
`src/cpu_device.cpp` and `src/kernels/`. CUDA, OpenCL, and HIP are optional
device layers built behind the same seam. Adding a device means adding a
kernel table, not restructuring the interpreter.

Load-time constant folding is implemented in `src/constant_folding.cpp`. A
node whose inputs are all initializers runs once during load with the CPU
kernels and becomes an initializer. The implementation reuses shape inference
on a one-node graph rather than maintaining a second execution path.

Shape inference and planning are a seam, not a load-time side effect. They take
concrete input shapes and return a plan. Re-planning for another shape does not
touch the parser, the IR, or any kernel. Symbolic dimensions load as `-1` and
are fixed for that load through full input shapes. Explicit non-positive
dimensions remain rejected.

A graph uses exactly one device, chosen at load. Per-operator host round-trips
are forbidden by design. CPU kernels live one file per operator under
`src/kernels/`; they stay naive and readable, and no vectorization or threading
is added without an approved specification.

The decision records that shape this architecture are:

- `D-4` — load-time constant folding uses the CPU kernels and reuses shape
  inference on a one-node graph.
- `D-5` — DOUBLE constants and value-info narrow to float32.
- `D-8` — symbolic dimensions load as `-1` and are fixed per load through full
  input shapes.
- `D-10` — BOOL constants and initializers load as bool.
- `D-15` — reductions accumulate in double and round once per output.

## Durable Sources of Truth

| Concern | Source |
| --- | --- |
| Mission and boundaries | `specs/mission.md` |
| Delivery order | `specs/roadmap.md` |
| Active work | `specs/YYYY-MM-DD-feature-name/requirements.md`, `plan.md`, and `validation.md` |
| Public version and history | `CHANGELOG.md` (0.2.0 unreleased) |
| What NERT is, how to build, test, and consume it | `README.md` |

## Deliberate Non-Choices

- NERT never depends on Neuriplo or any other consumer. The dependency arrow
  points one way; consumers adapt NERT, never the reverse.
- No per-operator device fallback. A graph either runs on its chosen device or
  is rejected at load.
- No CPU kernel optimization without an approved specification.
- No OpenCV, protobuf, or ONNX library in the runtime link closure.
- No framework or language migration to simplify a single feature.

## Validation Entry Points

The immutable acceptance path is the ResNet-18 fixture under CTest:

```bash
cmake -S . -B build
cmake --build build
python3 tools/provision_resnet18.py fixtures   # requires torch + torchvision 0.27.0
NERT_FIXTURE=fixtures/resnet18.onnx ctest --test-dir build --output-on-failure
scripts/format.sh --check
```

Parity against ONNX Runtime and the model-zoo harness live in Neuriplo
(`NertInferTest`, `nert_model_zoo_parity_test`). NERT-side specifications record
that evidence, but NERT never gates on exports it cannot provision itself.

## Decisions that Shape the Stack

These decisions were made on 2026-10-16 during the Phase 4 hardening pass.

- **`Q-1` — Opset policy.** Load declared ONNX-domain opsets 11 and above.
  Every source node records its effective schema from a full-history catalog
  generated by `python3 tools/gen_opset_compat.py`. The generator reads the
  reviewed per-operator implementation manifest in `tools/opset_manifest.py`;
  there is no canonical opset and no `BASELINE_OPSET` constant. The
  version-dependent forms are resolved by shape inference and the device
  rows: reduce axes, `ReduceSum` / `Squeeze` / `Unsqueeze` axes, Split
  `split`, legacy `Softmax` 2-D coercion, and `ScatterND.reduction`. There is
  no load-time canonicalization pass. The catalog gates every declaration
  against the implemented surface; guards include
  `ConvTranspose.output_shape`, `Resize.axes`, `Split.num_outputs`,
  `Cast.saturate`, `Cast.round_mode`, `Pad.wrap`, `AveragePool.dilations`,
  `Range.stash_type`, and all `GridSample` 20+ schemas until implemented.
  Models below 11 are converted with `onnx.version_converter` as documented
  in `README.md`. Revised 2026-10-22 for the version-aware packets.
- **`Q-3` — RAFT symbolic spatial dimensions.** Resolve them statically per
  load from the supplied input shapes, with no plan cache. Tier-4 RAFT parity
  (`max_scaled_diff=6.10e-05` on 2026-10-10) confirmed the choice.
- **`Q-4` — CI model subset.** CI carries only the ResNet-18 fixture,
  provisioned by `.github/workflows/ci.yml` through
  `tools/provision_resnet18.py`. The model zoo stays out of CI; parity runs
  live in Neuriplo.
- **`Q-5` — Public naming.** Namespace `nert`, headers under `include/nert/`,
  and lowercase `snake_case` C++ filenames, delivered as one breaking 0.2.0
  change. The Neuriplo adapter updates includes and namespace; the
  `neuriplo_engine` CMake alias remains for compatibility.
