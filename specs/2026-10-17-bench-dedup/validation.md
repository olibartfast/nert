# Validation — Bench every kernel, test every kernel, duplicate nothing

Defined before implementation. Bars: default OFF build untouched,
format clean, bench stays opt-in and out of the `nert` link closure.

## Checks

- [V-1] Slice 1: no `bench/gemm`, `bench/matmul`, or `bench/utils`
  remnant; OFF 8/8; CUDA bench leg builds and runs; format clean.
- [V-2] Slice 2: bench fixtures run per enabled device; numbers in
  `bench/README.md`.
- [V-3] Slice 3: shared helper header in use; legs green; line count down.
- [V-4] Slice 4: per-row unit-test + bench-fixture checklist complete.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| Slice 1 | `bench/gemm,matmul,utils` deleted + flags dropped + README rewritten; OFF 8/8 + CUDA bench 11 pass/1 SKIP (Conv, no cuDNN) + format | pass | 2026-10-17 | `bench/` is now `runtime` only; numbers match README magnitude. |
| — | — | pending | — | — |
