# Plan — Phase 5 Breaking 0.2 Rename

Branch `wip/nert-phase5` from main. Mechanical only.

## M1 — Move + rename (no content edits)

- `git mv include/engine include/nert`, then `git mv` each header to
  `snake_case`; same for `src/*.cpp`, `src/WireReader.hpp`,
  `src/kernels/*`, `test/*.cpp` (mapping in requirements [R-1]/[R-3]).
- Commit the moves separately (pure renames, no edits) for review.

## M2 — Identifier + reference sweep

- `sed`: `namespace engine` → `namespace nert`; `engine::` → `nert::`;
  `TEST(Engine` → `TEST(Nert`; `gtest_filter=Engine` → `=Nert`;
  `engine_` (CMake targets/tests) → `nert_`; `include/engine/` →
  `include/nert/`; per-file `#include "Kernels.hpp"` →
  `#include "kernels.hpp"` (mechanical fallout of M1).
- CMake source lists, `install(DIRECTORY include/nert)`,
  `scripts/format.sh` EXCLUDE paths, `project(... VERSION 0.2.0)`.
- Build + fix fallout until green.

## M3 — Docs + validation

- README, mission ([Q-5]), tech-stack ([Q-5], entrypoints `nert_*`),
  roadmap Phase 5 → Complete, CHANGELOG 0.2.0, this packet's
  `validation.md` evidence. Historical packets untouched.
- Full non-fixture ctest + format check; `git diff -w --stat` scope
  review (identifiers/paths only); merge; compact.
