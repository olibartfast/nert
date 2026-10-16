# Requirements — Phase 5 Breaking 0.2 Rename

Date: 2026-10-16. Roadmap: Phase 5 (was blocked on maintainer — authorized
this session). Constitution: mission/tech-stack/roadmap.

## Goal

One breaking commit: `include/engine/` + namespace `engine` → `nert`,
all C++ filenames → lowercase `snake_case` `.hpp`/`.cpp`, version 0.2.0.

## In Scope

- [R-1] `include/engine/` → `include/nert/`; headers renamed:
  `constant_folding.hpp`, `device.hpp`, `engine.hpp`, `executor.hpp`,
  `graph.hpp`, `model_loader.hpp`, `plan.hpp`, `shapes.hpp`.
- [R-2] `namespace engine` → `namespace nert` in all sources/tests;
  `engine::` → `nert::`; gtest suites `Engine*` → `Nert*`; ctest names
  and executables `engine_*` → `nert_*`.
- [R-3] Filenames → `snake_case`: `src/*.cpp` (`model_loader.cpp`,
  `shape_inference.cpp`, …), `src/kernels/*` (`mat_mul.cpp`,
  `gather_elements.cpp`, `instance_normalization.cpp`, `top_k.cpp`,
  `cpu_kernels.cpp`, `kernels.hpp`, …), `test/*_test.cpp`. `bench/`
  already compliant, untouched.
- [R-4] Build graph updated: root `CMakeLists.txt` sources +
  `install(DIRECTORY include/nert)`, `test/CMakeLists.txt` sources +
  target/test renames, `scripts/format.sh` EXCLUDE paths.
- [R-5] Living docs updated: README include + layout, mission/tech-stack
  ([Q-5] decided), roadmap Phase 5, CHANGELOG 0.2.0 header, validation
  entrypoints (`nert_*` suites). Historical packets untouched.
- [R-6] Version 0.1.0 → 0.2.0 (`project()`, CHANGELOG). CMake alias
  `neuriplo_engine` kept (zero cost; adapter change is includes +
  namespace, noted for Neuriplo).

## Out of Scope

- Any behavior change: no kernel/loader/shape/executor semantics edits;
  no new ops; no N1 work. Pure rename — `git diff -w` review must show
  only identifier/path churn.

## Constraints

- `git mv` for renames (history preserved). Single breaking commit.
- Class names (`Engine`, `CpuDevice`, …) unchanged — only the namespace.
