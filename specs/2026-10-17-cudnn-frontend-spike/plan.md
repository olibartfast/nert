# Plan — cuDNN Frontend spike

One worktree (`wip/nert-fespike`) holding the packet and any spike
notes. The spike program itself lives in `/tmp` (scratch, never
committed), like every earlier gate runner.

## Steps

1. Clone the frontend at the newest release tag into `/tmp`
   (not `develop`).
2. Scratch program: build + autotune a 3x3 conv graph and a YOLO-shape
   conv graph; run both plus the equivalent legacy-row calls from the
   worktree lib; print max diff and timings.
3. Record the table in `validation.md` and write the verdict.

## Validation

Evidence lands in
`specs/2026-10-17-cudnn-frontend-spike/validation.md` — adopt or drop.
