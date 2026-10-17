# Plan — Planner speed + resident GPU weights

Two slices, each merged on full evidence. One worktree per slice
(`wip/nert-plumb-<n>`).

## Slice A — planner jump + overflow guard (R-1, R-2)

- Replace the step-by-64 loop with jump-to-furthest-conflicting-end.
- Guard the element-count multiply; throw `ModelLoadException`
  naming the tensor before overflowing.
- Regression test: crafted residual graph, exact expected offsets +
  arena size (pins the identical-plan property).

## Slice B — resident initializers + RAII staging (R-3, R-4, R-5)

- Constructor uploads non-empty initializers once (non-host devices
  only; host path untouched); destructor releases.
- `Execute` binds initializer views to the resident buffers and
  stages graph inputs per run behind an RAII guard.
- Tests with a counting fake device (no GPU needed): initializer
  traffic happens once across two runs; a throwing kernel still
  releases staged inputs.

## Validation

Per-slice evidence in
`specs/2026-10-17-review-plumbing/validation.md`.
