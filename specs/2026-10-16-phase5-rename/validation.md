# Validation — Phase 5 Breaking 0.2 Rename

Defined before implementation. Evidence after execution.

## Checks

- [V-1] No `engine` namespace/dir remains in code: `grep -rn
  'namespace engine\|engine::\|include/engine/' src include test bench
  tools CMakeLists.txt test/CMakeLists.txt scripts/format.sh` empty
  (prose/history in specs/ + CHANGELOG exempt).
- [V-2] Filenames: `find src include test -name '*.cpp' -o -name
  '*.hpp' | grep -E '[A-Z]'` empty.
- [V-3] Build + non-fixture ctest 8/8 (renamed `nert_*` suites) +
  `scripts/format.sh --check` clean.
- [V-4] Scope: `git diff -w` shows identifier/path churn only — no
  kernel/loader/shape/executor semantic diffs.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| — | — | — | — | _to fill_ |
| V-1 | `grep -rn 'namespace engine\|engine::\|include/engine/' src include test bench tools CMakeLists.txt test/CMakeLists.txt scripts/format.sh` | empty (code) | 2026-10-16 | prose/history in specs/ + CHANGELOG exempt. |
| V-2 | `find src include test \( -name '*.cpp' -o -name '*.hpp' \) \| grep -E '[A-Z]'` | empty | 2026-10-16 | all `snake_case` (`mat_mul.cpp`, `gather_elements.cpp`, `top_k.cpp`, `is_nan.cpp`, `scatter_nd.cpp`, …). |
| V-3 | non-fixture ctest 8/8 (`nert_*`) + `scripts/format.sh --check` | pass | 2026-10-16 | fixture suites unrun (no fixture on box); EXCLUDE paths remapped. |
| V-4 | `git diff -w --stat` scope review | identifiers/paths only | 2026-10-16 | M1 pure moves (`5b9fe18`, 97 files, +0/-0); no kernel/loader/shape/executor semantic diffs. |
