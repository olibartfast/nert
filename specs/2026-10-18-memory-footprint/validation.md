# Validation — Memory Footprint Report

This validation is defined before implementation. Byte counts are exact; there
is no floating-point tolerance. Checks that depend on a GPU record `SKIP` with
the reason when the device is absent. CPU checks never skip.

## Acceptance Rules

- Every byte category in the report must match the accounting rules exactly.
- The report path must not allocate arena or device memory, transfer data, or
  dispatch kernels.
- A model is not considered implemented until the command-line tool and its
  documentation are verified against the built binary.
- Existing planner behavior, including offsets and `arena_size`, must remain
  unchanged.

## Exact Accounting on Hand-Built Graphs

Build a `Graph`, `InferredShapes`, and `MemoryPlan` directly and pass them to
`BuildMemoryFootprint`.

For an Add graph where `x` and initializer `c` are both `[2,4]` float32
(32 bytes each) and output `y` is `[2,4]`:

| Field | Host (`CpuDevice`) | Non-host fake device |
| --- | ---: | ---: |
| `initializer_bytes` | 32 | 32 |
| `input_bytes` | 32 | 32 |
| `output_bytes` | 32 | 32 |
| `arena_bytes` | 64 | 64 |
| `arena_payload_bytes` | 32 | 32 |
| `arena_peak_live_bytes` | 32 | 32 |
| `device_resident_bytes` | 0 | 32 |
| `host_model_bytes` | 96 | 32 |
| `device_model_bytes` | 0 | 96 |
| `run_peak_host_bytes` | 160 | 96 |
| `run_peak_device_bytes` | 0 | 128 |

For a three-link Relu chain with `x` of shape `[16]` float32 (64 bytes) and
output `y`:

| Field | Host | Non-host fake device |
| --- | ---: | ---: |
| `arena_bytes` | 128 | 128 |
| `arena_payload_bytes` | 192 | 192 |
| `arena_peak_live_bytes` | 128 | 128 |
| `input_bytes` | 64 | 64 |
| `output_bytes` | 64 | 64 |
| `initializer_bytes` | 0 | 0 |
| `device_resident_bytes` | 0 | 0 |
| `host_model_bytes` | 128 | 0 |
| `device_model_bytes` | 0 | 128 |
| `run_peak_host_bytes` | 256 | 128 |
| `run_peak_device_bytes` | 0 | 192 |

Also verify that an empty initializer counts as zero bytes; that a scalar
(`dims {}`) counts as one element; that a name which is both an input and an
initializer appears in both `I` and `In`; and that a missing inferred shape, a
negative dimension, or an unknown dtype throws `ModelLoadException` naming the
tensor.

## Planner Peak Semantics

Run `PlanMemory` on hand-built graphs and verify:

- A three-link chain: `arena_size=128`, payload 192, `peak_live_bytes=128`.
- A never-consumed tensor: `arena_size=64`, payload 128, peak 64.
- A graph output held through the last node: `arena_size=128`, payload 128,
  peak 128. If the output were released at its definition, the peak would be
  64.
- A zero-byte tensor: the buffer exists with size 0, and neither `arena_size`
  nor the peak changes.
- The existing exact-offset regression `a@0`, `b@1024`, `c@1088` with
  `arena_size=2112` is unchanged.
- `peak_live_bytes <= arena_size` on the ResNet-18 fixture plan.

## No Device Allocation in the Report Path

Use a fake non-host device with a counting allocator, transfer implementation,
and kernel table. `PlanMemoryFootprint` and `BuildMemoryFootprint` must leave
every counter at zero. `Model::memory_footprint()` must add no allocation or
transfer after construction. Calling it twice and calling it after `Run` must
leave the report unchanged.

## Model Parity

For a constructed `Model`, `model.memory_footprint()` must equal
`BuildMemoryFootprint(model.graph(), model.shapes(), model.plan(), device)`
field by field. Test this on the CPU device and on a fake non-host device.

## Real Models

- ResNet-18 fixture with `input=1,3,224,224`: record `initializer_bytes`,
  `arena_bytes`, `arena_payload_bytes`, `arena_peak_live_bytes`,
  `host_model_bytes`, `run_peak_host_bytes`, and the complete CLI output.
- A larger transformer export: use OWLv2 when the local export is available,
  otherwise the largest available opset-18 export. Record the same fields plus
  the export path and input names. This catches real dtypes, int64 shape
  constants, and multi-output graphs that hermetic tests do not.
- Sanity-check every model: `arena_peak_live_bytes <= arena_bytes`; record
  whether `arena_bytes < arena_payload_bytes`, which indicates arena reuse;
  and confirm `initializer_bytes` is nonzero for both models.

## Command-Line Tool

- `nert_mem_report --help` exits 0 and prints the usage, key list, and the
  NERT-managed-only limitation.
- The ResNet fixture command exits 0, prints every key in the required order
  exactly once, and reports `allocation=not-attempted`. The required order is
  defined by requirement `R-5`. With `--allocate` on CPU it reports
  `allocation=ok`.
- Missing `--input` for a graph input exits 1 with the shape-inference error on
  stderr and no partial report.
- A duplicate, unknown, or malformed `--input`, an unknown `--device`, or a
  non-positive dimension exits 2.
- `--device cuda`, `--device opencl`, and `--device hip` in a build without
  that layer print a `SKIP:` line and exit 0.
- On a CUDA build, plan-only mode prints a report. With `--allocate` on a host
  with an RTX 3060, it prints `allocation=ok` and records the bytes.
- On OpenCL and HIP builds, record either the report or the `SKIP` reason.
- If an allocation failure can be reproduced, for example with an absurd input
  size on a small device, the tool prints the plan-only report followed by
  `allocation=failed` and exits 1.

## Regression and Compatibility

- The default build passes the full CTest suite with the ResNet fixture.
- CUDA, OpenCL, and HIP builds pass their full suites or report the same clean
  skips as before.
- `scripts/format.sh --check` is clean.
- The diff contains only the paths listed in [`plan.md`](plan.md) plus this
  specification. No device kernel, loader, shape-inference, constant-folding,
  or bench source changes.
- Planner offsets and `arena_size` are identical before and after the change.
  The review-plumbing exact-offset test and the fixture plan pin this.

## Documentation

- `docs/memory-footprint-profiling.md` contains the accounting model, the
  planned CLI and API, the current-API workaround, measurement recipes, and the
  exclusions. Its status banner matches the implementation state, and every
  command and key it shows is verified against the built tool before the
  banner is removed.
- The guide's current-API workaround snippet compiles against the tree and
  prints the documented ResNet-18 totals. Links from `README.md` and the guide
  resolve.
- `README.md` links the guide under Documentation, and the opt-in flags table
  lists `NERT_BUILD_MEM_REPORT`.
- `CHANGELOG.md` contains the 0.2.0 Added entry for the API and tool.
- `specs/roadmap.md` moves the item from planned to complete with the
  validation evidence date.

## Acceptance Commands

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNERT_BUILD_MEM_REPORT=ON
cmake --build build --parallel "$(nproc)"
NERT_FIXTURE=/tmp/opencode/resnet18_export/resnet18.onnx \
    ctest --test-dir build --output-on-failure
./build/tools/mem_report/nert_mem_report \
    /tmp/opencode/resnet18_export/resnet18.onnx --input input=1,3,224,224
./scripts/format.sh --check
```

For a device build, configure a separate directory with the matching
`-DNERT_ENABLE_*` option and run the same tool with `--device cuda`,
`--device opencl`, or `--device hip`.

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
    -DNERT_ENABLE_CUDA=ON -DNERT_BUILD_MEM_REPORT=ON
cmake --build build-cuda --parallel "$(nproc)"
ctest --test-dir build-cuda --output-on-failure
./build-cuda/tools/mem_report/nert_mem_report \
    /tmp/opencode/resnet18_export/resnet18.onnx --input input=1,3,224,224 \
    --device cuda --allocate
```

Repeat with `NERT_ENABLE_OPENCL` and `NERT_ENABLE_HIP` where the toolkit
exists. Missing devices must record `SKIP` rather than a pass.

## Evidence

| ID | Check | Result | Date | Notes |
| --- | --- | --- | --- | --- |
| `V-1` | Exact accounting on hand-built graphs | Pending | — | — |
| `V-2` | Planner peak cases and fixture bound | Pending | — | — |
| `V-3` | Zero-allocation fake device | Pending | — | — |
| `V-4` | `Model` parity, repeated report, and report after `Run` | Pending | — | — |
| `V-5` | ResNet-18 and larger-export numbers | Pending | — | — |
| `V-6` | CLI help, errors, and device variants | Pending | — | — |
| `V-7` | Default and device CTest runs, format, diff audit | Pending | — | — |
| `V-8` | README, CHANGELOG, roadmap, and guide review | Partial | 2026-10-18 | The guide is written and linked. Its current-API snippet compiled and matched the documented ResNet-18 totals, and relative links resolve. The planned API and CLI remain unverified until implementation; CHANGELOG and roadmap completion are pending. |

## Deviations

None yet. Record any plan change here with the reason and date. A requirement
change updates `requirements.md` and this file together.
