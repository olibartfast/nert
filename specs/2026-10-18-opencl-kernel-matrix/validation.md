# Validation — OpenCL kernel matrix (Phase 1)

Rows land as OPS only; model cells become PASS only from a full-model
reference gate. Tolerance: `ExpectNearOcl` unless exact.

## Checks

- [V-M1] MatMul: 2-D hand-computed, dot product, 1-D promotion both
  ways, two batch-broadcast shapes, negative contraction/broadcast/
  output, CPU-oracle compare.
- [V-M2] LayerNormalization: 2x3 scale+bias hand-computed, axis=1 and
  scalar scale, off-default epsilon, negative axis/scale, CPU-oracle
  compare.
- [V-M3] Table registration, CMake, per-row test targets, format.
- [V-D1] OFF full `ctest`. [V-D2] CUDA-ON full `ctest`.
  [V-D3] OpenCL-ON full `ctest`. [V-D4] cuDNN farm full `ctest`.
  [V-D5] format clean. [V-D6] diff audit.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-M1 | `./build-ocl/test/nert_opencl_mat_mul_test` | pass 7/7 | 2026-10-19 | {22,28,49,64}; dot 32; [2,1,2,3]x[3,2] -> [2,1,2,2]; [2,2,3]x[3,2] -> [2,2,2]; 3 negatives. |
| V-M2 | `./build-ocl/test/nert_opencl_layer_norm_test` | pass 4/4 | 2026-10-19 | 2x3 scale/bias, axis=1 + scalar scale, epsilon 1e-3, bad axis/scale. |
| V-M3 | Table 21→23, CMake source/test registration | pass | 2026-10-19 | `MatMul`, `LayerNormalization` rows; separate per-row test targets. |
| V-D1 | OFF leg | pass 12/12 | 2026-10-19 | |
| V-D2 | CUDA-ON leg | pass 24/24 | 2026-10-19 | |
| V-D3 | OpenCL-ON leg | pass 16/16 | 2026-10-19 | 14 prior + MatMul + LayerNorm. |
| V-D4 | cuDNN farm leg | pass 24/24 | 2026-10-19 | `LD_LIBRARY_PATH` as recorded in coverage. |
| V-D5 | `scripts/format.sh --check` | pass | 2026-10-19 | `--fix` on integration edits. |
| V-M4 | OpenCL executor arena: per-offset buffers for handle-based devices + two-output model regression | pass 5/5 device tests | 2026-10-19 | Before: `arena_ + offset` on a `cl_mem` segfaulted in the driver at the first non-zero-offset tensor. |
| V-M5 | ResNet-18 full-model OpenCL gate (224x224, 1000 classes) vs ORT | pass | 2026-10-19 | top-1 107 = ORT; logits maxdiff 3.695e-06, mean 7.944e-07, 54 ms; OpenCL vs CPU max 2.168e-03. |
| V-M6 | Expand: 7 cases (rows/columns/scalar/identity copy/int64/bool/negatives) | pass 7/7 | 2026-10-19 | Cache-key collision between dtype entry points found and fixed (`ExpandF/I/B`). |
| V-M7 | Where: mask/broadcast/int64 + 3 negatives | pass 4/4 | 2026-10-19 | Lane kernel needed its launch locals renamed (TensorView shadowing). |
| V-M8 | OWLv2 row batch: Cast, ConstantOfShape, Equal/Greater/LessOrEqual/And, IsNaN/Erf/Elu, ReduceL2/ArgMax, two Einsum equations | pass 9/9 tests | 2026-10-19 | Table 25→37; einsum supports `...pd,...qd->...pq` and `bchw,bnc->bnhw` only, with a named error otherwise. |
| V-M9 | OWLv2 full OpenCL gate vs ORT: logits exact, top-1 3600/3600, pred_boxes max 5.093e-04, 0 over tolerance | pass | 2026-10-19 | 952-node model, 4 s; runner uses the same zero-pixel/ids fixture as the CPU/CUDA gate. |
| V-M10 | Grounding DINO CPU/CUDA reference gate | blocked | 2026-10-19 | Both devices fail at `/model/model/Clip`: "tensors must be float32" (int64 token ids). ORT reference: logits [1,900,256], pred_boxes [1,900,4]. Requires an int64 Clip variant; not claimed. |
| V-D6 | Diff audit | pass | 2026-10-19 | Kernel/test files + executor + shared registration + coverage/CHANGELOG. |

## Deviations

- Both lane implementers hit their step budget: Lane B (MatMul) left
  complete kernel/test files without shared-file registration; Lane C
  (LayerNormalization) left a kernel with a broken `SetArg` block and no
  test. The integration owner fixed the launch block (by-value args,
  correct scale/bias slots) and wrote the LayerNorm test.
- `MatMul` and `LayerNormalization` accumulate in float32 while the CPU
  oracle uses double; comparisons use the 1e-3 scaled tolerance.
- The first full OpenCL model exposed a real executor bug: arena views were
  built as `base + offset`, which is only valid for pointer devices. OpenCL
  now allocates one buffer per planned offset; the two-output regression test
  covers the path.
- The MatMul build emits a GCC `-Wmaybe-uninitialized` note from
  `std::variant` internals in the test; it is not in shared headers,
  the build is not `-Werror`, and the test passes.
