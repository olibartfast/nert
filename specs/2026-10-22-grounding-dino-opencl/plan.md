# Plan — Grounding DINO OpenCL live-buffer remedy

## T-1 — Lifetime metadata

Record each planned arena tensor's defining node and inclusive last consumer in
`Model`. Graph outputs keep the final node as their last use. Pointer-offset
devices ignore this metadata and keep the single-arena allocation.

## T-2 — Handle-path execution

For `!supports_arena_offsets()` devices:

- allocate a node output at its defining node with the planned tensor size;
- release buffers whose last use is before the current node;
- reuse freed buffers best-fit, with a 256 MiB cache cap so the pool cannot
  recreate the construction-demand blow-up;
- release every survivor after graph-output download;
- release on kernel failure through the per-run buffer guard.

## T-3 — Tests

Add a handle-shaped test device and a three-node chain test. Prove one resident
initializer, no construction arena, four allocations/releases per run, and
balanced destruction across repeated runs.

## T-4 — Gate

Rebuild `build-ocl`, relink the generic gate runner, and run Grounding DINO on
OpenCL. Record exit status, wall time, executed behavior, peak device memory,
and output comparison against the existing ORT reference and CPU/CUDA results.

## T-5 — Documentation

Close the Grounding DINO OpenCL cell only if the gate evidence passes. Record
the kernel/ABI policy and update the release coverage, inventory, and changelog.
