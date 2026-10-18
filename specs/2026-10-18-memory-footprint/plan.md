# Plan — Memory Footprint Report

## Working Constraints

Work on a feature branch off `main`. This change touches only the CPU-side
planner and report surface. Do not modify the CUDA, OpenCL, or HIP device
layers, the loader, shape inference, constant folding, or any kernel.

This plan implements requirements `R-1` through `R-7`: report contents
(`R-1`), accounting rules (`R-2`), planner peak (`R-3`), public API (`R-4`),
command-line tool (`R-5`), compatibility (`R-6`), and documentation (`R-7`).
It is validated by checks `V-1` through `V-8` in
[`validation.md`](validation.md); `V-6` covers the CLI key set.

## Phase 1 — Library Report Core

The first phase delivers a library-only report that a caller can obtain from a
loaded graph before allocating, plus the `Model` accessor.

### Planner peak

- Add `int64_t peak_live_bytes = 0;` to `MemoryPlan` in
  `include/nert/plan.hpp`, with the inclusive-liveness semantics defined in the
  requirements.
- In `src/memory_planner.cpp`, after the `tensors` vector is built, sweep
  events over `graph.nodes.size()`: add each tensor's size at its definition,
  subtract it after its last use when `end + 1 < nodes.size()`, then prefix-sum
  and take the maximum. Use the existing `ArenaTensor::live` interval; do not
  recompute liveness.
- Do not change placement, offsets, buffer sizes, or `arena_size`.
- Extend `test/memory_planner_test.cpp` with these cases:
  - a three-link chain: `arena_size=128`, `arena_payload=192`,
    `peak_live_bytes=128` for 16-float tensors of 64 bytes each;
  - an unused tensor: `arena_size=64`, payload 128, peak 64;
  - a graph output held to the last node: `arena_size=128`, payload 128,
    peak 128, proving that outputs are not released early;
  - a zero-byte tensor: the buffer exists with size 0 and the peak is
    unaffected;
  - the existing exact-offset regression (`a@0`, `b@1024`, `c@1088`, arena
    2112) still passes unchanged.

### Report value and builders

- Create `include/nert/memory_footprint.hpp` with the `MemoryFootprint` struct
  and the two builders described in the requirements.
- Create `src/memory_footprint.cpp` with:
  - `ElementBytes(DataType)` for float32, int64, and bool, returning zero for
    other types;
  - `ElementCount(dims)` that treats a scalar (`{}`) as one element and
    supports zero dimensions;
  - checked `int64` addition that throws `ModelLoadException` naming the
    category;
  - an initializer sum, input and output sums from `InferredShapes`, the
    arena payload total, and `device_resident_bytes` mirroring
    `UploadResidents` (`is_host`, zero-byte, known dtype, present values);
  - totals from the requirements' accounting rules;
  - no device allocator, transfer, kernel, or device mutation. The device is
    used only for `name()` and `is_host()`.
- Add `src/memory_footprint.cpp` to the `nert` target in `CMakeLists.txt`.
- Create `test/memory_footprint_test.cpp` with the `NertMemoryFootprint.*`
  suite and add its executable and CTest entries to `test/CMakeLists.txt`.
  Cover:
  - an Add graph with a 32-byte initializer: the host report has
    `arena=64`, `payload=32`, `peak=32`, `host_model=96`, and `run_host=160`;
    the non-host report has `resident=32`, `device_model=96`,
    `run_device=128`, `host_model=32`, and `run_host=96`;
  - a three-link chain: `arena=128`, `payload=192`, `peak=128`,
    `host_model=128`, `run_host=256`; on a non-host device,
    `run_device=192` and `run_host=128`;
  - empty-tensor and scalar cases;
  - a fake non-host device with a counting allocator, proving that
    `BuildMemoryFootprint` and `PlanMemoryFootprint` allocate zero bytes;
  - a name that is both an input and an initializer, counted in both
    categories, with the actual allocations documented;
  - overflow of `arena_payload_bytes`, which throws `ModelLoadException`
    naming the category.

### Model accessor

- Include `nert/memory_footprint.hpp` from `include/nert/executor.hpp` and
  declare `MemoryFootprint memory_footprint() const;`.
- Return `BuildMemoryFootprint(graph_, shapes_, plan_, *device_)` from
  `src/executor.cpp`.
- In `test/memory_footprint_test.cpp` or `test/executor_test.cpp`, verify that
  a constructed model's report equals the free-function report; that calling
  the accessor twice, and after `Run`, leaves the report unchanged; and that
  requesting the report does not move the construction allocation counters.

### Phase acceptance

A clean configure and build with the default options passes, the new suites
pass, the existing plan and executor suites still pass, and formatting is
clean. No GPU is required.

## Phase 2 — Command-Line Tool

### Target and argument handling

- Add `tools/mem_report/main.cpp` and `tools/mem_report/CMakeLists.txt`.
- In the root `CMakeLists.txt`, add
  `option(NERT_BUILD_MEM_REPORT "Build the nert_mem_report tool" OFF)` and
  include `tools/mem_report` when the option is enabled. Add the subdirectory
  after the test and bench subdirectories so `NERT_FIXTURE` is available when
  tests are enabled.
- Parse a positional model path, repeated `--input NAME=D0,D1,...`,
  `--device`, `--allocate`, and `--help` exactly as specified in the
  requirements.
- Device selection is compiled-in only. Define `NERT_MEM_REPORT_HAVE_CUDA`,
  `NERT_MEM_REPORT_HAVE_OPENCL`, and `NERT_MEM_REPORT_HAVE_HIP` to mirror the
  `NERT_ENABLE_*` options. A requested layer that is not compiled in prints
  the `SKIP:` line and exits 0.
- The plan-only path is `LoadGraphFromFile`, input parsing, and
  `PlanMemoryFootprint`. The `--allocate` path constructs a `Model` and calls
  `memory_footprint()`. If construction fails, print the plan-only report,
  then `allocation=failed`, and exit 1.
- Emit the stable keys in the required order and write errors to stderr.

### CLI tests

- Add a CTest test `nert_mem_report_help` that runs `--help` and expects exit
  status 0.
- When `NERT_BUILD_TESTS` and `NERT_FIXTURE` are set, add
  `nert_mem_report_fixture`, which runs the tool on the ResNet-18 fixture with
  `--input input=1,3,224,224` and uses `PASS_REGULAR_EXPRESSION` on
  `arena_bytes=[0-9]+`. The full key set is checked in the validation phase.
- Add a `WILL_FAIL` test for the fixture without `--input` because shape
  inference must fail, and a usage test for an invalid device name.
- Manual validation covers the device variants and `--allocate`.

### Phase acceptance

A clean configure with `-DNERT_BUILD_MEM_REPORT=ON` builds the tool. The help
and fixture tests pass. The ResNet-18 command prints the full key set. The tool
links only `nert`, so the runtime link closure does not change.

## Phase 3 — Evidence, Documentation, and Synchronization

- Run the validation matrix in [`validation.md`](validation.md). Record the
  exact ResNet-18 numbers and at least one larger transformer export: OWLv2
  when the local export is available, otherwise the largest available
  opset-18 export, with its path recorded.
- Run CPU and every available device build (CUDA, OpenCL, HIP). Record `SKIP`
  with the reason when a device is absent.
- Update `docs/memory-footprint-profiling.md` by removing the planned banner,
  verifying every command and output key against the built tool, and
  refreshing the example. Add the documentation link and the
  `NERT_BUILD_MEM_REPORT` row to `README.md`. Add the 0.2.0 entry to
  `CHANGELOG.md`. Move the roadmap item from planned to complete once the
  implementation lands.
- Keep the diff to the paths listed below plus this specification.

## Integration Checkpoints

- After the library phase, a caller can build synthetic graphs and get exact
  reports; `Model::memory_footprint()` matches; all existing tests pass.
- After the CLI phase, a clean `-DNERT_BUILD_MEM_REPORT=ON` build produces a
  working `nert_mem_report` from the repository root.
- After the evidence phase, the evidence table is complete, the documentation
  is accurate, and the roadmap and changelog are synchronized.

## Expected Touched Paths

```text
include/nert/plan.hpp
include/nert/memory_footprint.hpp          (new)
include/nert/executor.hpp
src/memory_planner.cpp
src/memory_footprint.cpp                   (new)
src/executor.cpp
test/memory_planner_test.cpp
test/memory_footprint_test.cpp             (new)
test/CMakeLists.txt
CMakeLists.txt
tools/mem_report/CMakeLists.txt            (new)
tools/mem_report/main.cpp                  (new)
docs/memory-footprint-profiling.md
README.md
CHANGELOG.md
specs/roadmap.md
specs/2026-10-18-memory-footprint/         (this specification)
```

No device kernel, loader, shape-inference, constant-folding, or bench source
changes are expected.

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

For a device build, add the matching `-DNERT_ENABLE_*` flag, use a separate
build directory, and run the same tool with `--device cuda`, `--device opencl`,
or `--device hip`. Use `--allocate` only where the device is present.
