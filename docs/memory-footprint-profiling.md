# Memory footprint profiling

Status: **planned**. The `MemoryFootprint` API and `nert_mem_report` CLI
are specified in
[`specs/2026-10-18-memory-footprint/`](../specs/2026-10-18-memory-footprint/)
and are not implemented in this revision. The accounting model and the
current-API workaround below work today; the planned commands are marked
as such so nobody mistakes them for shipped behavior.

## What this answers

The footprint report answers one question:

> For these concrete input shapes, how many bytes does NERT plan to manage
> on the selected device, and how do those bytes split into weights, the
> arena, graph I/O, and device-resident copies?

It is a **model fit/sizing** report. It is not RSS, not a device-memory
query, and not a hard guarantee that a model fits a device. Kernel
workspaces, driver contexts, allocator metadata, memory pools,
fragmentation, and container overhead are outside the account. See
[What the report does not count](#what-the-report-does-not-count).

## At a glance

| Task | Today | After the packet lands |
| --- | --- | --- |
| Arena, payload, I/O, initializer totals | current-API workaround below | `nert_mem_report` or `PlanMemoryFootprint` |
| Report for a constructed model | `model.plan()` for the arena; workaround for other totals | `model.memory_footprint()` |
| Peak live arena bytes | not exposed | `arena_peak_live_bytes` |
| Device-resident copies | sum initializer bytes for a non-host model | `device_resident_bytes` |
| Measured RSS or device usage | `/usr/bin/time -v`, `nvidia-smi` | unchanged; the report does not measure |

## The accounting model

All categories are byte counts. Payload sums (`initializer_bytes`,
`input_bytes`, `output_bytes`, `arena_payload_bytes`, and
`arena_peak_live_bytes`) are unaligned; `arena_bytes` includes the
planner's 64-byte alignment.

| Field | Scope | Meaning |
| --- | --- | --- |
| `initializer_bytes` | host | Embedded constants in the loaded `Graph`, including empty constants (0 bytes). |
| `input_bytes` | host, per run | Declared graph inputs at the concrete input shapes. |
| `output_bytes` | host, per run | Declared graph outputs at the concrete shapes. |
| `arena_bytes` | selected device | `MemoryPlan::arena_size`: the one allocation the executor reserves. |
| `arena_payload_bytes` | selected device | Sum of every arena buffer's size, with no reuse. |
| `arena_peak_live_bytes` | selected device, planned | Maximum simultaneously live arena payload over the node schedule. |
| `device_resident_bytes` | device | Copies of initializers uploaded once by a non-host `Model`; 0 on CPU. |
| `host_model_bytes` | host | `initializer_bytes + (host ? arena_bytes : 0)`. |
| `device_model_bytes` | device | `host ? 0 : arena_bytes + device_resident_bytes`. |
| `run_peak_host_bytes` | host, per run | `initializer_bytes + input_bytes + output_bytes + (host ? arena_bytes : 0)`. |
| `run_peak_device_bytes` | device, per run | `host ? 0 : arena_bytes + device_resident_bytes + input_bytes`. |

`arena_peak_live_bytes` is a **plan-level** high-water mark, not a
runtime measurement. Graph outputs are live through the last node under
the planner's inclusive liveness rule, and a never-consumed non-output is
live only at its defining node.

### Overlaps are deliberate

Two categories can name the same tensor because two allocations really
exist:

- An input that is also an initializer counts in both `initializer_bytes`
  and `input_bytes`. The `Graph` keeps the embedded constant, and the
  executor stages the input for a run.
- An output that is arena-backed is included in `arena_bytes`; its
  `output_bytes` copy is still additional in `run_peak_host_bytes`
  because every run copies the output out to caller-owned memory (host
  vectors or device download buffers).
- On a non-host device, `initializer_bytes` is the host copy and
  `device_resident_bytes` is the device copy. Both exist; do not add
  `host_model_bytes` and `device_model_bytes` into one "total".

### Alignment and reuse

- `arena_bytes` is 64-byte aligned; `arena_payload_bytes` and
  `arena_peak_live_bytes` are unaligned payload sums.
- If `arena_bytes < arena_payload_bytes`, the planner reused bytes across
  non-overlapping lifetimes.
- `arena_peak_live_bytes <= arena_bytes` always holds for a plan produced
  by `PlanMemory`. If the arena is much larger than the peak, look at
  alignment padding and placement before assuming a bug.

## Planned CLI how-to

These commands are the specified interface, shown here so it can be
reviewed. The binary does not exist until the packet lands.

Build the opt-in tool:

```bash
cmake -S . -B build -DNERT_BUILD_MEM_REPORT=ON
cmake --build build --target nert_mem_report
```

Report the pinned ResNet-18 fixture on CPU:

```bash
./build/tools/mem_report/nert_mem_report \
    fixtures/resnet18.onnx --input input=1,3,224,224
```

Report a CUDA target without allocating, then prove the arena and resident
copies allocate:

```bash
./build/tools/mem_report/nert_mem_report \
    fixtures/resnet18.onnx --input input=1,3,224,224 --device cuda
./build/tools/mem_report/nert_mem_report \
    fixtures/resnet18.onnx --input input=1,3,224,224 --device cuda --allocate
```

`--device` accepts `cpu`, `cuda`, `opencl`, or `hip`. A layer that was not
compiled in prints `SKIP:` and exits 0. A compiled-in OpenCL or HIP layer
whose device object cannot be constructed also prints `SKIP:`. Plan-only
mode does not touch the device allocator; on a CUDA build it reports the
target layout even without a GPU, so treat that as a projection.
`--allocate` constructs the `Model` and is the fit check; it proves arena
and resident allocation, not a full `Run` (no input staging, no kernel
dispatch).

Output is one `key=value` per line, in this order:

```text
device_name=cpu
host_device=true
initializer_bytes=<bytes>
input_bytes=<bytes>
output_bytes=<bytes>
arena_bytes=<bytes>
arena_payload_bytes=<bytes>
arena_peak_live_bytes=<bytes>
device_resident_bytes=<bytes>
host_model_bytes=<bytes>
device_model_bytes=<bytes>
run_peak_host_bytes=<bytes>
run_peak_device_bytes=<bytes>
allocation=not-attempted
```

`allocation` is `not-attempted`, `ok`, or `failed`. On a failed
`--allocate`, the tool prints the plan-only report plus
`allocation=failed`, writes the error to stderr, and exits 1. Usage errors
exit 2; load, shape, plan, and allocation failures exit 1.

## Planned C++ API how-to

Plan without allocating:

```cpp
#include "nert/device.hpp"
#include "nert/memory_footprint.hpp"
#include "nert/model_loader.hpp"

const nert::Graph graph = nert::LoadGraphFromFile("model.onnx");
const nert::MemoryFootprint footprint = nert::PlanMemoryFootprint(
    graph, {{"input", {1, 3, 224, 224}}}, nert::CpuDevice());
```

Report the plan a model already holds:

```cpp
#include "nert/device.hpp"
#include "nert/executor.hpp"

nert::Model model(graph, {{"input", {1, 3, 224, 224}}}, nert::CudaDevice());
const nert::MemoryFootprint footprint = model.memory_footprint();
```

`BuildMemoryFootprint(graph, shapes, plan, device)` is the lower-level
form when you already have inferred shapes and a plan. All three report
paths use only `Device::name()` and `Device::is_host()`; they do not
allocate arena or device storage.

## Today: plan-level workaround

Until the packet lands, `PlanMemory` and `Model::plan()` already expose
the arena, and the `Graph`/`InferredShapes` expose the other categories.
This small program computes everything except `arena_peak_live_bytes`.
It is deliberately temporary; migrate to `PlanMemoryFootprint` when the
packet lands so the formulas live in one place. The snippet hardcodes the
ResNet fixture's `input` name and `1,3,224,224` shape; change both for
your model (shape inference rejects a missing input).

```cpp
#include "nert/graph.hpp"
#include "nert/model_loader.hpp"
#include "nert/plan.hpp"
#include "nert/shapes.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int64_t ElementBytes(nert::DataType dtype) {
    switch (dtype) {
    case nert::DataType::Float32: return 4;
    case nert::DataType::Int64: return 8;
    case nert::DataType::Bool: return 1;
    case nert::DataType::Unknown: return 0;
    }
    return 0;
}

int64_t ElementCount(const std::vector<int64_t>& dims) {
    int64_t count = 1;
    for (int64_t dim : dims) {
        count *= dim;
    }
    return count;
}

int64_t TensorBytes(const nert::TensorInfo& info) {
    return ElementCount(info.dims) * ElementBytes(info.dtype);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <model.onnx>\n";
        return 2;
    }
    const nert::Graph graph = nert::LoadGraphFromFile(argv[1]);
    const nert::InferredShapes shapes =
        nert::InferShapes(graph, {{"input", {1, 3, 224, 224}}});
    const nert::MemoryPlan plan = nert::PlanMemory(graph, shapes);

    int64_t initializer_bytes = 0;
    for (const auto& entry : graph.initializers) {
        initializer_bytes += TensorBytes({entry.second.dtype, entry.second.dims});
    }

    int64_t input_bytes = 0;
    for (const auto& entry : graph.inputs) {
        input_bytes += TensorBytes(shapes.tensors.at(entry.first));
    }

    int64_t output_bytes = 0;
    for (const auto& entry : graph.outputs) {
        output_bytes += TensorBytes(shapes.tensors.at(entry.first));
    }

    int64_t arena_payload_bytes = 0;
    for (const auto& entry : plan.buffers) {
        arena_payload_bytes += entry.second.size;
    }

    const bool host_device = true; // false for CUDA/OpenCL/HIP
    const int64_t resident_bytes = host_device ? 0 : initializer_bytes;

    std::cout << "arena_bytes=" << plan.arena_size << '\n'
              << "arena_payload_bytes=" << arena_payload_bytes << '\n'
              << "initializer_bytes=" << initializer_bytes << '\n'
              << "input_bytes=" << input_bytes << '\n'
              << "output_bytes=" << output_bytes << '\n'
              << "device_resident_bytes=" << resident_bytes << '\n'
              << "host_model_bytes="
              << initializer_bytes + (host_device ? plan.arena_size : 0) << '\n'
              << "device_model_bytes="
              << (host_device ? 0 : plan.arena_size + resident_bytes) << '\n'
              << "run_peak_host_bytes="
              << initializer_bytes + input_bytes + output_bytes +
                     (host_device ? plan.arena_size : 0)
              << '\n'
              << "run_peak_device_bytes="
              << (host_device ? 0 : plan.arena_size + resident_bytes + input_bytes)
              << '\n';
    return 0;
}
```

Build it against a CPU build of the library after configuring the
project. Provision the pinned fixture first if you have not already
(needs torch and torchvision 0.27.0):

```bash
python3 tools/provision_resnet18.py fixtures
g++ -std=c++17 -I include footprint_today.cpp build/libnert.a -o footprint_today
./footprint_today fixtures/resnet18.onnx
```

For the pinned ResNet-18 fixture and `input=1,3,224,224`, this workaround
prints these plan totals (recompute after any planner or fixture change;
the planned report adds `arena_peak_live_bytes`):

```text
arena_bytes=6422528
arena_payload_bytes=22988704
initializer_bytes=46738848
input_bytes=602112
output_bytes=4000
device_resident_bytes=0
host_model_bytes=53161376
device_model_bytes=0
run_peak_host_bytes=53767488
run_peak_device_bytes=0
```

Read that as roughly 46.7 million bytes of embedded constants (mostly
weights) plus a 6.4-million-byte arena on the host. The 23.0-million-byte
payload total is about 3.6 times the arena, so reuse dominates the
difference. A run adds the 0.6 MB input and 4 KB output copies. There is
no device half on CPU.

Limitations of the workaround:

- `arena_peak_live_bytes` is not exposed by today's `MemoryPlan`; do not
  re-derive liveness in a second implementation. The packet adds it to
  the planner so the report and plan cannot drift.
- `device_resident_bytes` is assumed to equal `initializer_bytes` on a
  non-host device because `UploadResidents` uploads every non-empty
  initializer once. The `host_device` flag must match the device you
  actually use.
- There is no CLI, no `SKIP` handling, and no allocation check; the
  planned tool owns those.

## What the report does not count

The report accounts for the allocations in NERT's model of a run:
initializer storage, the arena, device copies, and graph I/O at the
supplied shapes. It excludes:

- **Measured memory**: RSS, process high-water marks, device-memory
  queries, allocator instrumentation, and leaks.
- **Kernel workspaces**: cuDNN and cuBLAS scratch, OpenCL buffers created
  inside kernels, and any per-call temporary the kernel owns.
- **Runtime and driver overhead**: CUDA context, driver pools,
  fragmentation, allocator metadata, graph/container overhead, and the
  cost of the loaded `Graph` outside the initializer payload.
- **Caller-side extras**: storing the result, copies your application
  makes, and any input preprocessing buffers.

Because workspaces are missing, a model that reports comfortably under a
device limit can still fail to allocate or run. Use the report to compare
models, catch arena regressions, and size the NERT-managed part; use a
measured run for the hard fit decision. If `--allocate` fails while the
planned totals look under the limit, the gap is one of the excluded
items; the report cannot explain it. Reduce the input shape or measure
the actual process/device.

## Measuring what the report cannot

When you need the measured number, measure the actual process or device;
do not treat a plan total as one.

Linux peak RSS:

```bash
/usr/bin/time -v ./your_app model.onnx
# "Maximum resident set size (kbytes)" is the process peak, including
# the runtime, driver, workspaces, and your application.
```

Device memory while a run is in flight:

```bash
watch -n 0.5 'nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```

Record a baseline before the model loads (the CUDA context alone is
nontrivial), then subtract it from the peak. For CPU allocation profiles,
`heaptrack` or Valgrind Massif show where the bytes come from; neither is
part of NERT's build.

## Common mistakes

- Adding `host_model_bytes` and `device_model_bytes` into one total. They
  are different memory spaces.
- Treating `arena_peak_live_bytes` as RSS. It is a planned liveness
  high-water mark, not a measurement.
- Forgetting that outputs are copied out of the arena, so
  `run_peak_host_bytes` includes `output_bytes` even when the output is
  arena-backed.
- Counting initializer bytes only once on a GPU. The host `Graph` keeps
  them and the device gets a resident copy.
- Assuming `arena_bytes` equals the sum of tensor sizes. Reuse makes the
  arena smaller; alignment can make it larger.
- Comparing reports made with different input shapes without re-running
  shape inference and planning.

## Validation recipe

The packet's `validation.md` pins exact byte tables and the no-allocation
proof. Once the packet lands, check a change locally:

1. Run the planner tests and the report tests with the ResNet fixture:
   `NERT_FIXTURE=fixtures/resnet18.onnx ctest --test-dir build --output-on-failure`.
2. Assert `arena_peak_live_bytes <= arena_bytes` on every reported plan.
3. On a hand-built Add or chain graph, compare the report fields with the
   exact tables in `validation.md`; do not eyeball a real model for this.
4. Use a fake device with a counting allocator to prove the report path
   makes no device allocation.
5. Re-run the CLI on CPU, then on each available device leg, and record
   `SKIP` with the reason when a device is absent.

## Related

- Specification:
  [`specs/2026-10-18-memory-footprint/`](../specs/2026-10-18-memory-footprint/)
  (`requirements.md`, `plan.md`, `validation.md`).
- Build and test entrypoints: [`README.md`](../README.md).
- Arena planner and liveness rules: `include/nert/plan.hpp` and
  `src/memory_planner.cpp`.
