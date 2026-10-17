# Validation — Planner speed + resident GPU weights

Defined before implementation. Bars: OFF 8/8, format clean, no
behaviour change beyond the two fixes.

## Checks

- [V-1] Slice A: crafted-graph test pins exact offsets + arena size;
  overflow guard throws `ModelLoadException` naming the tensor.
- [V-2] Slice B: fake-device tests prove one initializer upload across
  two runs and zero leaked staged buffers on kernel throw; all
  existing executor/device legs green; CUDA bench numbers move in
  the faster direction (or stay SKIP-clean).

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| Slice A | Planner jump-to-end + overflow guards; exact-offset regression (a@0/b@1024/c@1088/arena 2112) + 2 overflow negatives; OFF 8/8 + format | pass | 2026-10-17 | 2 plan failures are missing-fixture (pre-existing, environmental). |
| — | — | pending | — | — |
