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
