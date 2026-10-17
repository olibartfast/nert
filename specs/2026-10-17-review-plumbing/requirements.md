# Requirements — Planner speed + resident GPU weights

Date: 2026-10-17. Source: external architecture review (verified
claim by claim against the tree on 2026-10-17; two nuances found and
recorded below). This packet covers review items 1 and 2 only. Item 3
(loader hardening) and item 4 (docs/API) are separate packets.

## Problem 1 — the memory planner is quadratic

`PlanMemory` finds a free offset by stepping 64 bytes at a time and
rechecking every placed buffer at each step. Measured on a synthetic
residual chain: 500 nodes / 3.1 MB tensors take 1.7 s, 1000 nodes take
6.6 s. Real attention graphs are bigger. In the same function,
`elements *= dim` can overflow `int64` before the `size < 0` check
runs (undefined behaviour; the check only catches wraparounds that
happen to land negative).

## Problem 2 — the GPU executor re-uploads every weight per run

On non-host devices each `Execute` allocates, synchronously uploads,
and later frees every initializer plus every input. For ResNet-18
that is ~46 MB of host-to-device traffic and ~60 malloc/free pairs
per inference — it dominates kernel speedups and makes GPU bench
numbers misleading. It also contradicts the header, which promises
the arena is allocated exactly once. In the same path, a throwing
kernel leaks the per-run staged buffers (no RAII guard).

## In scope

- [R-1] Planner jumps past the furthest-ending conflicting buffer
  instead of stepping by 64. The plan produced is exactly the same
  (no offset below that end can be free); a regression test pins
  exact offsets and the arena size on a crafted graph.
- [R-2] `elements *= dim` is guarded by an `INT64_MAX / dim` check
  that throws `ModelLoadException` before multiplying.
- [R-3] Initializers upload once in the `Model` constructor and live
  as members until destruction. Each `Run` stages only the graph
  inputs, ideally into preallocated slots so the hot path allocates
  nothing per run beyond inputs.
- [R-4] Per-run staged buffers sit behind an RAII guard, so a
  throwing kernel cannot leak them.
- [R-5] Behaviour is unchanged: exact CPU semantics for null/empty
  views, identical numerics, all existing tests green.

## Out of scope

- Loader hardening, docs/API cleanup (their own packets).
- Pinned host buffers and non-default streams (noted as later work).
- Changing the `Model` constructor signature or the borrowed-graph
  rule (review item 5, not this packet).

## Constraints

- No `src/kernels`/loader/shape-inference logic changes; planner and
  executor files only, plus tests.
- The upload-once proof must not need a GPU: a counting fake device
  in the test asserts one initializer upload across two runs and
  balanced allocate/release when a kernel throws.
- Format clean, OFF 8/8, device legs green.
