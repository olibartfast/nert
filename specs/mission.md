# NERT Mission

> Status: constitution for the standalone NERT repository, derived 2026-10-10
> from the neuriplo packets `2026-09-17-native-engine-cpu-spine`,
> `2026-10-01-yolo26-native`, and `2026-10-10-native-tasks-op-coverage`
> (tiers 1–3 landed; tiers 4–5 open). Implementation and executable checks
> take precedence if this file drifts.

## Why NERT Exists

NERT (Neuriplo Engine Runtime) is a first-party ONNX inference runtime: a CPU
reference interpreter for ONNX opset 18. It parses the graph, infers shapes
for the requested input, plans a single arena allocation, and executes nodes
one by one in topological order. It computes in float32, with int64/bool
shape data and I/O.

NERT exists so Neuriplo has a self-contained CPU backend with no vendor SDK
underneath, and so a future accelerated path has an executable correctness
oracle to be checked against — not a performance target.

## Who It Serves

- Neuriplo, which consumes NERT as its `NERT` backend via FetchContent pinned
  by `NERT_VERSION`, adapting it to `InferenceInterface`. NERT itself has no
  knowledge of Neuriplo.
- Maintainers adding operator coverage kernel by kernel, with hand-computed
  tests and parity against ONNX Runtime through Neuriplo's model-zoo harness.
- Future device layers (Phase N1), which plug into the device seam without
  touching the device-agnostic core.

## Product Promise

- A dependency-free runtime: C++ standard library only (tests use GoogleTest;
  fixture provisioning uses torch/torchvision, never the runtime).
- An interpreter, not a compiler: no graph lowering, fusion (beyond
  load-time constant folding), autotuning, or serialized engines.
- One device per loaded graph, chosen at load. A graph that cannot run
  entirely on the requested device is rejected at load; there is no per-op
  fallback at run time.
- CPU kernels that are deliberately naive, readable, and unoptimized — the
  correctness oracle, never a performance claim.
- Static shapes resolved at load from concrete input shapes. Load rejects
  unknown ops, unsupported attribute combinations, and unresolvable dims with
  node name and op type — never at first inference, never by guessing.

## Boundaries and Non-Goals

- NERT does not reimplement vendor SDKs for their own sake and never hides
  their constraints; vendor backends stay the reference NERT is measured
  against (in Neuriplo).
- No CUDA in the current scope (Phase N1 owns the GPU device layer).
- No quantization, FP16/INT8, batching beyond declared dims, or dynamic-shape
  runtime machinery beyond static per-load planning ([D-8] in tech-stack).
- No task pre/postprocessing, serving APIs, or product UI — those belong in
  Neuriplo, neuriplo-tasks, or consumers.
- NERT does not promise every ONNX op: coverage grows by surveyed exports,
  one tier at a time, each proven by parity.

## Engineering Values

Correctness and readability over speed. Direct, operational docs: supported
paths, exact commands, failure behavior, no implied coverage. Every
non-trivial change carries requirements → plan → validation, with validation
defined before code and evidence recorded after execution.

## Open Product Questions

- Quantitative latency/memory targets: none (CPU path is explicitly not a
  performance target).
- [Q-5] decided 2026-10-16 (Phase 5, breaking 0.2): `include/nert/` +
  namespace `nert`, all C++ filenames lowercase `snake_case`.
