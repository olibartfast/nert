# Memory Footprint Report

> Status: specified, not implemented. Date: 2026-10-18.

## Purpose

NERT (Neuriplo Engine Runtime) already plans the whole graph into one arena and
exposes `MemoryPlan::arena_size` plus per-buffer offsets. It does not expose the
rest of the footprint: embedded weights, graph inputs and outputs at concrete
shapes, device-resident initializer copies, or the peak simultaneously live
payload. Without those numbers, a caller cannot answer “how much memory will
this model need on this device?” without re-deriving planner internals.

This specification adds a deterministic, NERT-managed footprint report at plan
time, a `Model` accessor, and a thin command-line tool. The report shows the
bytes held in initializers, graph inputs and outputs, the planned arena and its
peak live payload, device-resident initializer copies, and the resulting model
and run totals. It is a sizing account, not a measurement of process or device
memory. It does not allocate arena or device storage in the plan-only path and
does not change planning or execution.

## What It Deliberately Excludes

This is an account of memory NERT manages. It is not proof that a model fits a
device. It excludes measured runtime memory (resident set size, process
high-water marks, allocator instrumentation, leak detection, and device-memory
queries), kernel-internal workspaces from cuDNN, cuBLAS, or OpenCL, CUDA
context and driver overhead, memory pools, fragmentation, allocator metadata,
per-tensor lifetime traces, arena visualisations, comparisons of alternative
plans, and any machine-readable output beyond the stable `key=value` lines
defined below.

Measured runtime memory would be a separate allocator-instrumentation effort
and is not scheduled. Neuriplo and `bench/` integration are also out of scope.

## Scope

The work includes:

- plan-level accounting through a new `MemoryPlan::peak_live_bytes` field,
  computed from the planner's existing inclusive liveness without changing
  offsets or `arena_size`;
- a public `MemoryFootprint` value with byte counts by category and documented
  totals, plus builders for plan-time and model-time use;
- a `Model::memory_footprint()` accessor that reports the plan the model
  actually holds;
- an opt-in `nert_mem_report` command-line tool that loads a model, accepts
  concrete input shapes, reports the footprint, and can optionally prove
  allocation;
- support for CPU and every compiled-in device layer (CUDA, OpenCL, and HIP);
  the report uses only `Device::name()` and `Device::is_host()`, so plan-only
  mode needs no allocation; a device layer that is not compiled in, or whose
  device object cannot be constructed, reports `SKIP`;
- tests for exact accounting, planner peak semantics, the no-allocation
  guarantee, `Model` parity, and the command-line tool.

## Report Contents

`include/nert/memory_footprint.hpp` defines:

```cpp
struct MemoryFootprint {
    std::string device_name;
    bool host_device = true;

    int64_t initializer_bytes = 0;       // embedded constants in the Graph (host)
    int64_t input_bytes = 0;             // declared graph inputs at concrete shapes
    int64_t output_bytes = 0;            // declared graph outputs at concrete shapes
    int64_t arena_bytes = 0;             // MemoryPlan::arena_size
    int64_t arena_payload_bytes = 0;     // sum of all arena buffer sizes
    int64_t arena_peak_live_bytes = 0;   // max simultaneously live arena payload
    int64_t device_resident_bytes = 0;   // device copies of initializers

    int64_t host_model_bytes = 0;
    int64_t device_model_bytes = 0;
    int64_t run_peak_host_bytes = 0;
    int64_t run_peak_device_bytes = 0;
};
```

All byte fields are payload bytes stored in `int64`. Empty tensors contribute
zero bytes. `device_name` is exactly `Device::name()`, and `host_device` is
exactly `Device::is_host()`.

## Accounting Model

Define the following values:

- `I = initializer_bytes`
- `In = input_bytes`
- `Out = output_bytes`
- `A = arena_bytes`
- `R = device_resident_bytes`
- `H = host_device`

The report must satisfy these rules.

**Initializers.** `I` is the sum, over every initializer, of its element count
times its element size. Empty initializers contribute zero. An initializer that
is also a declared graph input or output is still counted here.

**Graph inputs.** `In` is the sum of every declared graph input at its concrete
inferred shape. A name that is both an input and an initializer counts in both
`I` and `In`: the executor keeps both storages and stages the input on every
run.

**Graph outputs.** `Out` is the sum of every declared graph output at its
concrete inferred shape. Outputs are copied out of the arena on every run,
either into host result vectors or into device download buffers. `Out` is
therefore additional to `A` in the run peaks even when the output is
arena-backed.

**Arena size.** `A` is exactly `MemoryPlan::arena_size`.

**Arena payload.** `arena_payload_bytes` is the sum of `plan.buffers[*].size`
over every arena buffer, without reuse.

**Peak live payload.** `arena_peak_live_bytes` is the maximum, over node
indices, of the total size of arena tensors live at that node. It uses the
planner's inclusive `[definition, last_use]` intervals. Graph outputs are live
through the last node. A never-consumed non-output is live only at its defining
node. For a plan produced by `PlanMemory`, this value equals
`plan.peak_live_bytes` and must satisfy
`arena_peak_live_bytes <= arena_bytes`.

**Device residents.** `R` is the sum of the bytes that `UploadResidents`
actually uploads on a non-host device: non-empty initializers with a known
element type and present values. It is zero on a host device.

The derived totals are:

```text
host_model_bytes       = I + (H ? A : 0)
device_model_bytes     = H ? 0 : A + R
run_peak_host_bytes    = I + In + Out + (H ? A : 0)
run_peak_device_bytes  = H ? 0 : A + R + In
```

All sums are overflow-checked. An `int64` overflow throws
`ModelLoadException` naming the category. A missing inferred shape, a negative
dimension, or an unknown element type in a summed tensor throws
`ModelLoadException` naming the tensor.

## Planner Peak

`MemoryPlan` gains one field:

```cpp
int64_t peak_live_bytes = 0;
```

`PlanMemory` computes it from the `ArenaTensor` liveness it already builds. The
sweep adds each tensor's size at its definition, subtracts it after its last
use, and records the maximum prefix sum. It must not change any
`BufferAssignment` offset, any buffer size, or `arena_size`. The existing
exact-offset regression test remains the proof that placement is unchanged.

## Public API

`include/nert/memory_footprint.hpp` also defines:

```cpp
MemoryFootprint BuildMemoryFootprint(const Graph& graph,
                                     const InferredShapes& shapes,
                                     const MemoryPlan& plan,
                                     const Device& device);

MemoryFootprint PlanMemoryFootprint(const Graph& graph,
                                    const ShapeMap& input_dims,
                                    const Device& device);
```

`BuildMemoryFootprint` uses the supplied plan as-is. It does not re-plan,
re-infer shapes, mutate any argument, allocate arena or device storage,
transfer data, or dispatch a kernel. The device is used only for `name()` and
`is_host()`.

`PlanMemoryFootprint` is the convenience path: infer shapes, plan memory, then
build the report. It throws exactly what those functions throw and introduces
no new exception type.

`Model` gains:

```cpp
MemoryFootprint memory_footprint() const;
```

It returns `BuildMemoryFootprint(graph_, shapes_, plan_, *device_)`. It does not
run inference, allocate arena or device storage, or upload anything. A
constructed model's report is stable across `Run` and `RunTyped` calls and
equals the free-function report built from its `graph()`, `shapes()`, `plan()`,
and device.

The header is installed by the existing `install(DIRECTORY include/nert ...)`
rule and adds no dependency beyond the C++ standard library.

## Command-Line Tool

Add an opt-in target named `nert_mem_report`, controlled by
`NERT_BUILD_MEM_REPORT` (default `OFF`), with its source in
`tools/mem_report/main.cpp`.

```text
nert_mem_report <model.onnx> --input NAME=D0,D1,... [--input ...]
                [--device cpu|cuda|opencl|hip] [--allocate] [--help]
```

`--device` defaults to `cpu`. Every declared graph input must be supplied
exactly once. A duplicate, unknown, malformed, or non-positive dimension is a
usage error and exits with status 2.

Without `--allocate`, the tool is plan-only: it loads, infers, plans, and
reports without touching the device allocator. On a compiled-in CUDA build,
this reports the target layout even when no driver or GPU is present; that is a
projection, not a claim that the device is available. `--allocate` constructs a
`Model`, allocating the arena and uploading resident initializers, and reports
from `Model::memory_footprint()`. It is the presence and fit check. It does not
stage inputs or dispatch kernels, so it proves the arena and resident
allocations, not a full `Run`.

Output is stable, one `key=value` pair per line with no spaces, in this order:

```text
device_name
host_device
initializer_bytes
input_bytes
output_bytes
arena_bytes
arena_payload_bytes
arena_peak_live_bytes
device_resident_bytes
host_model_bytes
device_model_bytes
run_peak_host_bytes
run_peak_device_bytes
allocation
```

Booleans print as `true` or `false`. `allocation` is `not-attempted`, `ok`, or
`failed`.

If `--allocate` fails, print the plan-only report followed by
`allocation=failed`, write the error to stderr, and exit 1.

A requested device layer that is not compiled in prints
`SKIP: device '<name>' is not compiled in` and exits 0. A compiled-in device
whose object cannot be constructed prints `SKIP: <reason>` and exits 0,
matching the benchmark convention.

Load, shape-inference, planning, and allocation failures exit 1. Usage errors
exit 2. `--help` exits 0 and states the NERT-managed-only limitation: no RSS,
no kernel workspaces, and no device-fit guarantee. This specification includes
no JSON or CSV mode.

## Compatibility and Constraints

- Standard library only; C++17; no new runtime or tool dependency.
- The design follows `specs/tech-stack.md` and `specs/mission.md`: a C++17
  static library with public headers in `include/nert/`, a standard-library-only
  runtime, one device per graph chosen at load, static shapes per load, and no
  CPU-kernel optimization.
- The new `MemoryPlan` field is additive; existing aggregate initialisations and
  consumers keep compiling.
- No change to the parser, shape inference, constant folding, kernels, executor
  behavior, planner placement, or device layers beyond the additive peak field
  and report accessor.
- Existing tests are not modified to accommodate the change and stay passing;
  no test is weakened or skipped. New tests cover the new behavior.
- `scripts/format.sh --check` stays clean.

## Documentation Deliverables

- `docs/memory-footprint-profiling.md` is the user guide: the accounting model,
  the planned CLI and API, the current-API workaround, measurement recipes, and
  the exclusions. It carries a `planned` status banner until the implementation
  lands.
- `README.md` links the guide under Documentation and marks it planned. When
  the feature is implemented, the guide's banner and future-tense command
  language are removed, the README opt-in flags table gains
  `NERT_BUILD_MEM_REPORT`, and the guide's example output is refreshed.
- `CHANGELOG.md` records the added API and tool under 0.2.0.
- `specs/roadmap.md` lists this work as planned until it lands, then as complete
  with evidence.

## Traceability

| ID | Topic | Section |
| --- | --- | --- |
| `R-1` | Report shape and field semantics | [Report Contents](#report-contents) |
| `R-2` | Accounting rules and derived totals | [Accounting Model](#accounting-model) |
| `R-3` | Planner peak exposure | [Planner Peak](#planner-peak) |
| `R-4` | Public API and `Model` accessor | [Public API](#public-api) |
| `R-5` | Command-line tool | [Command-Line Tool](#command-line-tool) |
| `R-6` | Constraints and compatibility | [Compatibility and Constraints](#compatibility-and-constraints) |
| `R-7` | Documentation deliverables | [Documentation Deliverables](#documentation-deliverables) |

## Decisions

The maintainer interview on 2026-10-18 settled the scope:

- The primary use case is model fit and sizing by category, plus peak live
  bytes.
- The surface is a public C++ API on `Model` plus a thin command-line tool.
- Devices are CPU plus every compiled-in device layer, with `SKIP` when a layer
  is absent.
- The deliverable for that session was the specification packet only.

Decisions recorded by the planner:

- `--allocate` is opt-in so a fit preflight must work without touching a
  device allocator or risking an out-of-memory failure.
- Kernel workspaces and measured RSS stay out of scope; `README.md` and the
  tool's `--help` state the limitation.
- The tool uses stable `key=value` lines rather than JSON so no schema or
  formatting dependency enters the runtime.

No open questions remain.
