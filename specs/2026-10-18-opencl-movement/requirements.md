# Requirements — OpenCL shared movement rows

Date: 2026-10-18. Per `specs/2026-10-17-task-models/coverage.md`
(Hermes full-matrix pass): CUDA is op-complete — no more CUDA rows.
The portable gap is OpenCL/HIP at 9/72 rows, with `Conv` (5 models)
the hard dependency no full model can run without. The useful order
is shared movement/reduction rows first (independent, reused), then
pools, then `Conv`, then transformer-specific rows. This packet lands
the top shared-movement set on OpenCL, in coverage.md order:
`Transpose` (4 models/252 touches), `Slice` (4/85), `Concat` (4/83),
`Shape` (4/41), `Unsqueeze` (4/37), `Gather` (4/22).

## In scope

- Six hand-written OpenCL C kernels (`src/opencl/`, runtime-compiled
  via `opencl_runtime`, first-party like the 9 existing rows), table
  rows in `src/opencl/table.cpp`, `target_sources` entries, tests in
  `test/opencl_kernels_test.cpp` (hand-computed + oracle-compare +
  negatives, `GTEST_SKIP` without an OpenCL GPU platform).
- Oracle policies mirrored exactly (per-op notes below); no new policy.
  Rank cap 16 on packed-index rows (established pilot limit).

## Out of scope

- HIP mirrors (no hipify toolchain on this host; Q-1), OpenCL `Conv`
  (own slice after movement+pools), pools (queued next), `bench/`
  fixtures, loader/shape/executor changes, Neuriplo-side runs.

## Oracle policies (binding)

- `Transpose`: float32/int64/bool, `perm` attr (default reverse);
  per-output gather via aligned strides.
- `Slice`: float32/int64/bool data; int64 starts/ends (+ optional
  axes/steps) inputs downloaded to host; negative wrap + clamp per
  oracle.
- `Concat`: float32/int64, shared dtype+rank, axis wrap; per-block
  D2D copies (ConcatCopy precedent) — no kernel needed beyond
  copies, like the CUDA row.
- `Shape`: int64 out; `start`/`end` int attrs, negative wrap, clamp.
- `Unsqueeze`: float32/int64/bool; int64 axes input (negative wrap,
  sorted-unique, range check); buffer copy, shape owned by inference.
- `Gather`: float32/int64/bool data + int64 indices; `axis` attr
  default 0 with wrap; negative-index wrap; per-output gather.

## Constraints (traced to constitution)

- stdlib-only default; OpenCL PRIVATE and opt-in. One device per
  graph, no fallback. Naive readable kernels, `clFinish` sync like
  the existing rows (Relu precedent).

## Open questions

- None: scope (movement six) and order (coverage.md) decided
  2026-10-18 (maintainer-directed). Pools next, then Conv.
