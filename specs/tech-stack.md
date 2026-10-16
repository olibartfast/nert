# NERT Technical Stack

> Status: constitution, derived 2026-10-10 from neuriplo `specs/tech-stack.md`
> (First-Party NERT section) and the three native-engine packets. The
> implementation and executable checks take precedence on drift.

## Core Stack

| Area | Choice | Boundary |
| --- | --- | --- |
| Library | C++17 static library `nert` (alias `nert::nert`, compat alias `neuriplo_engine`) | Do not raise the standard without an explicit decision. Public headers: `include/nert/` (namespace `nert` since 0.2). All C++ filenames lowercase `snake_case` `.hpp`/`.cpp`. |
| Build | CMake ≥ 3.16 | `cmake -S . -B build && cmake --build build`. Tests on when top-level (`NERT_BUILD_TESTS`, default ON top-level / OFF when consumed; `BUILD_INFERENCE_ENGINE_TESTS` still honoured for Neuriplo). |
| Required libraries | C++ standard library only | No new runtime dependency without explicit approval. Tests: GoogleTest. Fixture provisioning: torch + torchvision 0.27.0 (tooling only, never linked). |
| ONNX | opset 18, ONNX domain only; hand-written protobuf wire reader | Loader targets opset 18. Other opsets convert with `onnx.version_converter` for validation ([Q-1] open: accept natively or require 18 + conversion note). |
| Compute | float32 compute; int64/bool shape data and graph I/O | Reductions accumulate in double, round once per output ([D-15]). DOUBLE constants/value-info narrow to float32 ([D-5]). BOOL constants/initializers load ([D-10]). |
| Tests | GoogleTest + CTest, suites `nert_*` (renamed from `engine_*` in 0.2) | Fixture: opset-18 ResNet-18 via `tools/provision_resnet18.py`, selected by `NERT_FIXTURE`. |
| Quality | clang-format (check `scripts/format.sh --check`) | Keep the checked-in script scope; CI pins clang-format 18. |

## Architecture (binding)

- Interpreter: `src/ModelLoader.cpp` (parse) → `src/ShapeInference.cpp`
  (shapes) → `src/MemoryPlanner.cpp` (arena) → `src/Executor.cpp`
  (sequential dispatch) → `src/kernels/` (CPU table behind `src/CpuDevice.cpp`).
- IR: `include/engine/Graph.hpp` — node list in topological order, typed
  tensor value table (float32/int64/bool), initializer storage with dtype
  tags, explicit graph inputs / initializers / intermediates.
- Device seam (`include/engine/Device.hpp`): core is device-agnostic;
  allocator, transfer, and op-keyed kernel table are per-device. CPU is the
  only layer. Adding a device = adding a kernel table, not a refactor.
- Load-time constant folding (`src/ConstantFolding.cpp`, [D-4]): a node whose
  inputs are all initializers runs once at load with the CPU kernels and
  becomes an initializer. Reuses shape inference on a one-node graph.
- Shape seam: shape inference and the planner take concrete input shapes and
  return a plan — never one-time load side effects. Re-planning for another
  shape touches no parser, IR, or kernel. Symbolic dims load as -1 and are
  fixed per load through full input shapes ([D-8]); explicit non-positive
  dim values stay rejected.
- One device per graph, chosen at load; per-op host round-trips forbidden by
  design.
- CPU kernels: one file per op under `src/kernels/`, naive and readable, no
  vectorization/threading without an approved packet.

## Durable Sources of Truth

| Concern | Source |
| --- | --- |
| Mission and boundaries | `specs/mission.md` |
| Delivery order | `specs/roadmap.md` |
| Active work | `specs/YYYY-MM-DD-feature-name/{requirements,plan,validation}.md` |
| Public version and history | `CHANGELOG.md` (0.1.0 unreleased) |
| What NERT is / build / test / consume | `README.md` |

## Explicit Non-Choices

- No dependency from NERT on Neuriplo (or any consumer) — the arrow points
  one way; Neuriplo adapts NERT, never the reverse.
- No per-operator device fallback.
- No CPU kernel optimization without an approved packet.
- No OpenCV, protobuf, or ONNX library in the runtime link closure.
- No framework/language migration to simplify one feature.

## Validation Entrypoints (immutable acceptance = ResNet fixture ctest)

```bash
cmake -S . -B build
cmake --build build
python3 tools/provision_resnet18.py fixtures   # needs torch + torchvision 0.27.0
NERT_FIXTURE=fixtures/resnet18.onnx ctest --test-dir build --output-on-failure
scripts/format.sh --check
```

Parity against ONNX Runtime and the model-zoo harness live in Neuriplo
(`NertInferTest`, `nert_model_zoo_parity_test`); NERT-side packets record
their evidence there but never gate on exports NERT cannot provision itself.

## Open Technical Decisions

Decided 2026-10-16 (Phase 4):

- [Q-1] Opset policy: **require opset-18 exports + conversion note**
  (was: recommendation). Loader accepts opset 18 only; README documents
  the `onnx.version_converter` command. No native 12–17 support.
- [Q-3] RAFT symbolic spatial dims: **static per load** via input shapes
  ([D-8]) — confirmed by Tier-4 RAFT parity (`max_scaled_diff=6.10e-05`,
  2026-10-10). No plan cache.
- [Q-4] CI model subset: **ResNet-18 fixture only** (provisioned by
  `.github/workflows/ci.yml` via `tools/provision_resnet18.py`). The
  model zoo stays out of CI; parity runs live in Neuriplo
  (`NertInferTest`, `nert_model_zoo_parity_test`).
- [Q-5] Decided 2026-10-16 (Phase 5, breaking 0.2.0): namespace `nert`,
  `include/nert/`, all C++ filenames lowercase `snake_case`. Neuriplo
  adapter change: includes + namespace (CMake alias `neuriplo_engine`
  kept).
