# Validation — Grounding DINO OpenCL live-buffer remedy

Defined before implementation. The gate is the measured full-model leg, not a
modelled footprint.

## Checks

- [x] [V-1] Handle-device lifetime belongs to [R-1]/[R-2]: a handle-shaped test
  device allocates only live intermediates, reuses freed buffers under the
  256 MiB cap, releases everything on a normal run and on a kernel throw, and
  leaves pointer-offset devices on the single-arena path. Verified 2026-10-22
  by `NertExecutor.HandleDeviceAllocatesOnlyLiveTensors` and
  `NertExecutor.HandleDeviceReleasesBuffersOnKernelThrow`; CPU CTest 13/13.
- [x] [V-2] Grounding DINO OpenCL leg belongs to [R-5]: full run exits 0, writes
  both outputs, stays under the device limit, and agrees with the accepted
  CPU/CUDA task-equivalence evidence. Verified 2026-10-22: exit 0, 19,224
  nodes, wall 38:27.88, peak board 2,215 MiB; canonicalized pred_boxes
  0/3,600 (max 7.045e-05), IoU min 0.999759 with 900/900 at or above 0.999,
  top-1 token 900/900, decoded set_diff 0. The canonicalized logits envelope
  mean is 7.63878e-03, the same diagnostic breach already recorded for CUDA
  (not one of the task criteria).
- [x] [V-3] No ABI change belongs to [R-4]: `Model`, `TensorView`, `Device`,
  and kernel signatures are unchanged; the slot map is retained for layout
  compatibility but is no longer populated by the handle path. Format check
  298 files.

## Evidence

| Check | Command | Result | Date | Notes |
|---|---|---|---|---|
| V-1 | `ctest --test-dir build --output-on-failure` | pass (13/13) | 2026-10-22 | Two new handle-device tests; existing CPU/CUDA arena tests unchanged. |
| V-2 | `/tmp/opencode/nert_gate_opencl_final ... groundingdino-18.onnx opencl ...` | pass | 2026-10-22 | Exit 0; 19,224 nodes; wall 38:27.88; peak board 2,215 MiB; both outputs written; canonicalized task gate 0/3,600 boxes, IoU min 0.999759 (900/900), top-1 900/900, set_diff 0. |
| V-3 | `scripts/format.sh --check`; header/layout review | pass | 2026-10-22 | 298 files formatted; no public ABI change. |
