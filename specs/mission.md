# NERT Mission

> Status: constitution for the standalone NERT repository. Derived on
> 2026-10-10 from the Neuriplo packets `2026-09-17-native-engine-cpu-spine`,
> `2026-10-01-yolo26-native`, and `2026-10-10-native-tasks-op-coverage`.
> If this document and the implementation disagree, the implementation and
> its executable checks win.

## What NERT Is

NERT (Neuriplo Engine Runtime) is a first-party inference runtime for models in
the Open Neural Network Exchange (ONNX) format. It reads a model, resolves
static shapes for the requested inputs, plans one arena for intermediate
tensors, and executes the graph node by node. Compute is float32; int64 and
bool are used for shapes, indices, masks, and graph inputs or outputs.

The project is deliberately small in scope. It is a CPU reference interpreter,
not a compiler, a serving stack, or a performance product. It exists so
Neuriplo has a self-contained backend with no vendor SDK underneath, and so
accelerated device layers have an executable correctness oracle to measure
against.

## Who It Serves

- **Neuriplo** consumes NERT as its `NERT` backend through a pinned
  `FetchContent` dependency and adapts it to `InferenceInterface`. NERT has no
  knowledge of Neuriplo.
- **Maintainers** add operator coverage one kernel at a time, with
  hand-computed tests and parity runs against ONNX Runtime through
  Neuriplo's model-zoo harness.
- **Device-layer authors** plug new backends into the device seam without
  changing the device-agnostic core.

## Product Commitments

- **No runtime dependencies.** The library uses the C++ standard library only.
  Tests use GoogleTest; fixture provisioning uses PyTorch and torchvision as
  tooling, never as runtime links.
- **An interpreter, not a compiler.** NERT does not lower, fuse (beyond
  load-time constant folding), autotune, or emit serialized engines.
- **One device per graph.** The device is chosen when the model is loaded. A
  graph that cannot run entirely on that device is rejected at load; there is
  no per-operator fallback at inference time.
- **A readable CPU oracle.** CPU kernels are intentionally naive and
  unoptimized. They define correctness; they are not a performance claim.
- **Static shapes resolved at load.** The loader rejects unknown operators,
  unsupported attribute combinations, and unresolvable dimensions, naming the
  node and operator. It never defers these failures to the first inference or
  guesses a shape.
- **Failures are explicit.** Unsupported paths are rejected with a useful
  message rather than silently approximated.

## Boundaries and Non-Goals

- NERT does not reimplement vendor SDKs for their own sake, and it does not
  hide vendor constraints. Vendor libraries remain the reference against which
  NERT is measured in Neuriplo.
- The CPU interpreter is the required path. GPU device layers are opt-in
  extensions delivered under Phase N1; they share the same device seam and do
  not change the core contract.
- No quantization, FP16 or INT8 compute, batching beyond declared dimensions,
  or dynamic-shape machinery beyond static per-load planning (decision `D-8`).
- No task pre/postprocessing, serving API, or product UI. Those belong in
  Neuriplo, `neuriplo-tasks`, or the consumer.
- NERT does not promise every ONNX operator. Coverage grows from surveyed
  exports, one operator family at a time, and each addition is proven by
  parity.

## Engineering Principles

Correctness and readability come before speed. Documentation should state the
supported path, the exact command, and the failure behavior without implying
coverage the code does not provide. Every non-trivial change carries
requirements, a plan, and validation; validation is defined before the code is
written and evidence is recorded after it runs.

## Open Questions and Decisions

- No quantitative latency or memory targets exist. The CPU path is explicitly
  not a performance target.
- The public API lives under `include/nert/` in namespace `nert`, with
  lowercase `snake_case` filenames. This was decided in the Phase 5 rename and
  is recorded as `[Q-5]` in the specification history.
