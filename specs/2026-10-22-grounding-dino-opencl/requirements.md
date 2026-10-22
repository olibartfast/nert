# Requirements — Grounding DINO OpenCL live-buffer remedy

Date: 2026-10-22. Supersedes the assessment-only remedy note in
`specs/2026-10-19-catalog-program/handoff-boundary-4.md`; the maintainer has
authorized the bounded Option A implementation.

## Problem

The OpenCL device uses opaque `cl_mem` handles, so the executor cannot add a
byte offset to a base arena allocation. The handle path currently materializes
one buffer per distinct planned offset at Model construction: Grounding DINO
demands 11,028.5 MiB of construction-time buffers on a 5,803.3 MiB device even
though the planner's live-set peak is 760.35 MiB (1,687.9 MiB including weights
and staged inputs). The run fails with `CL_MEM_OBJECT_ALLOCATION_FAILURE`.

## Goal

Make handle-based devices allocate only tensors that are live at a node and
release each one after its inclusive last use, without changing the public ABI,
the kernel ABI, or pointer-offset device behavior.

## In Scope

- [R-1] Handle devices allocate a node output at its defining node and release
  it after its last consumer, except graph outputs, which survive until the
  output download.
- [R-2] Pointer-offset devices (CPU/CUDA) keep the single-arena allocation and
  its existing behavior.
- [R-3] Release all live buffers on kernel failure and at the end of a Run.
- [R-4] Add an executor test that proves per-run allocation/release counts and
  repeated-run balance on a handle-shaped device.
- [R-5] Rebuild the OpenCL engine and rerun the Grounding DINO OpenCL leg; the
  leg must complete and its outputs must agree with the existing CPU/CUDA
  task-equivalence evidence within the documented deviation.

## Out of Scope

- The kernel-offset arena (public `TensorView.data` offset channel).
- Kernel math, OpenCL program changes, or a public ABI change.
- The six missing-export catalog entries.

## Constraints

- C++17, standard library only.
- No public header layout change to `TensorView` or device interfaces.
- Existing CPU/CUDA/OpenCL full-model gates stay unchanged.
