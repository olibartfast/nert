# Release gates — 2026-10-24

Base: `95027c3`; changes are local and uncommitted. Recovered OpenCode session:
`ses_e95444623ffegPuDxHToNSWm7S`. AMD cloud validation is excluded by the user;
its full-catalog/CI coverage is now accepted as known missing for 0.2.0 by
[a subsequent maintainer decision](known-missing-amd.md). The signed review and
acceptance JSON retain the earlier policy state; their test results are unchanged.

## Runtime fixes

- Split schema 18 `num_outputs`: loader accepts and validates the attribute;
  shape inference, CPU, CUDA, HIP and OpenCL use the same partition rule.
  Uneven final chunks work. Counts, conflicting explicit sizes, old-schema
  misuse and zero/negative final extents and output counts exceeding the axis dimension fail explicitly. Legacy equal splitting
  and the shape-inference exception type are preserved.
- Sign: dtype-preserving float32/int64 rows on all four backends. Comparison
  rather than subtraction handles INT64_MIN, signed zero and infinities; NaN is explicitly propagated as ONNX requires.
  The generated schema catalog and all four registries contain the operator.
- CTest: shape and planner suite filters now match `EngineShapes*` and
  `EnginePlan*`; the old `NertShapes*`/`NertPlan*` filters selected zero cases.
- `tools/check_model_outputs.py` fails on numeric mismatches, non-finite
  values, missing outputs, incorrect byte counts and missing references.
  The recovered report-only comparator always exited zero and is not used
  as acceptance evidence.

## Available exports and acceptance

All five available exports load and dispatch fully on CPU/CUDA/OpenCL after
Split/Sign support. RF-DETR gates use the provisioned `-sim.onnx` artifacts,
with new ORT 1.27.0 CPUExecutionProvider references for those exact artifacts.
Simplification is an exporter option, not proof of raw-export equivalence:
the preserved export notes describe numerical differences. Unsimplified
RF-DETR still needs static shape-expression coverage and is not claimed here.

The strict criterion remains `abs(diff) <= 1e-3*(1+abs(reference))`, exact for
integer outputs; every value must be finite. The approved **separate** task
criterion is implemented by `check_task_equivalence.py`: threshold 0.25,
identical retained query set and classes, strict numerical checks for every
retained output element, and exact retained binary masks at logit >= 0.
Empty detection sets cannot pass vacuously. Raw mismatches remain recorded.
The user approved this criterion on 2026-10-24 ("go"). Raw failures remain
part of the accepted, fixture-specific deviation; this is not a general
claim of raw-output equality or other-input equivalence. Independent review passed; the final preserved-evidence acceptance result is
recorded in `evidence/acceptance.json`.

- EdgeCrafter detection: CPU/CUDA swap rows 295/296, reference confidence
  0.01949710 / 0.01949695; all nine retained detections agree. OpenCL raw
  passes. No TopK rule or tie-break was changed.
- EdgeCrafter segmentation: CPU raw passes. CUDA has one raw mask element
  over tolerance; OpenCL has 634, all in background query 166 (score 0.02672).
  The ten retained detections and their masks pass. All 300 binarized
  OpenCL masks are exactly equal to the reference.
- EdgeCrafter pose: CUDA/OpenCL raw pass; CPU has low-score query ordering
  differences. Retained-query acceptance is measured separately.
- RF-DETR detection-small: CPU/CUDA/OpenCL raw pass.
- RF-DETR segmentation-medium: CPU/CUDA/OpenCL raw fail, but all six retained
  detections pass, including every retained mask element and binary pixel.
  Raw masks over tolerance: 18,229 CPU / 12,309 CUDA / 26,162 OpenCL; all-query mask IoU
  minima are recorded per leg. CPU also passes the retained-output criterion.

The OpenCL segmentation diagnostic probed the pre-mask contraction inputs.
Replacing its contraction with a double computation on identical device
inputs did not remove the upstream query differences. No speculative
accumulation or TopK changes were made.

## Unavailable catalog entry

Entry 24, “RF-DETR keypoint other sizes”, has no distinct upstream model.
The installed `rfdetr==1.8.0` exposes only `RFDETRKeypointPreview`, already
covered by entry 7; the sibling exporter's named-size options silently fall
back to Preview. Counting that fallback as a new model would be false.
Upstream [variants](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/variants.py)
and [keypoint documentation](https://github.com/roboflow/rf-detr/blob/develop/docs/learn/run/keypoints.md)
also list Preview only (checked 2026-10-24). The user approved deferring this unavailable, duplicate placeholder on
2026-10-24. It is explicitly deferred until a distinct upstream size ships,
not marked PASS; entry 7 continues to cover Preview. There are now 29
in-scope entries, two original exclusions, and one deferred placeholder.

## Reproduction

Scratch root: `/tmp/nert-release-gates/`. Each model directory contains its
frozen input `.bin`/`.npy`, `ref_*.npy`, device raw outputs, full run logs,
strict metrics and approved task metrics. The small JSON evidence and export
provenance are preserved in `evidence/`; models and arrays stay outside Git.

For example:

```bash
python3 tools/check_model_outputs.py /tmp/nert-release-gates/ecseg_s \
  /tmp/nert-release-gates/ecseg_s/opencl
# Expected exit 1: strict raw mismatch is preserved.
python3 specs/2026-10-24-release-gates/check_task_equivalence.py \
  /tmp/nert-release-gates ecseg_s opencl
# Expected exit 0: approved retained-detection criterion.
```

No release tag, GitHub release or remote change has been made.

## Validation summary

Final rebuilt CPU CTest 14/14, CUDA/cuDNN CTest 27/27, and OpenCL CTest
32/32 pass. The corrected shape/planner filters execute actual tests.
HIP/NVIDIA has 28/28 existing kernel cases plus the three new Sign/Split
regressions passing with no skips. The generated catalog check (290 entries),
clang-format-18 check (308 files), and strict-comparator failure-path smoke
checks pass. Native AMD and its persistent cloud CI were not run.

Full-model gate binaries were freshly linked during this packet; the final
source adjustments after linking restore ModelLoadException on invalid
Split shapes, reject invalid Split partitions with non-positive final chunks,
and preserve Sign NaNs. The frozen graphs use valid Split counts and int64
Sign operands, so these adjustments leave their successful calculations
unchanged. Review verified the only Sign operands: EdgeCrafter det/seg
`getitem_188`, int64 `[1,300]`, ranges 0..23946 / 0..23978. The other three
models have no Sign nodes. Final unit/device suites were rebuilt after those adjustments. Existing older catalog/model and
Docker evidence was inherited, not rerun wholesale; the five newly available
models were rerun on all three local backends.

All **15/15** full-model task-equivalence checks pass on the frozen fixture.
Every model process exited zero and dispatched all nodes: EdgeCrafter
1,523 / 1,569 / 1,534; RF-DETR simplified 844 / 1,069. Raw comparison fails
on eight of those legs and remains explicitly recorded. The seven strict
passes and eight approved fixture-specific task passes must not be conflated.
The release scoreboard is 29/29 available in-scope entries, with the named
keypoint-size placeholder explicitly deferred. Independent review passed for this packet; native AMD remains outside this task.

`evidence/artifacts.json` pins model/external-weight, input, reference, raw
output, executable and harness-source hashes. It also distinguishes model
process exit from numeric gate status; exit zero alone is never a parity pass.

### Exact commands and preserved harness

`evidence/commands.json` contains all 15 successful process invocations as
argument arrays, their working directory and CUDA library environment.
`evidence/gate_runner.cpp` is an exact copy of the scratch source used to link
the recorded binaries (including its original scratch-only header comment).
No new model export was run in this packet: existing artifacts were recovered
and hashed. Historical RF export driver descriptions are provenance, not a
claim that those export commands were executed again.

The actual linking commands, run from `/home/oli/repos/nert`, were:

```bash
g++ -std=c++17 -O2 -I include /tmp/opencode/nert_gate.cpp build/libnert.a -lpthread -o /tmp/nert-release-gates/gate_cpu
g++ -std=c++17 -O2 -DNERT_GATE_ENABLE_OPENCL -I include /tmp/opencode/nert_gate.cpp build-ocl/libnert.a -lOpenCL -lpthread -o /tmp/nert-release-gates/gate_opencl
g++ -std=c++17 -O2 -DNERT_GATE_ENABLE_CUDA -I include /tmp/opencode/nert_gate.cpp build-cudnn/libnert.a -L/home/oli/.local/lib/python3.12/site-packages/nvidia/cu13/lib -lcudart -lcublas -L/tmp/opencode/cudnn/lib64 -lcudnn -lpthread -o /tmp/nert-release-gates/gate_cuda
```

For reproduction, the preserved identical source may replace
`/tmp/opencode/nert_gate.cpp`. `provision_refs.py` recreates the frozen inputs
and references using the recorded artifacts, with no downloads. Its 2026-10-24
rerun is checked against the previously recorded hashes before accepting the
packet. The full-model driver and strict per-element comparison are preserved
as `run_model_gates.py`; supply `cpu`, `cuda` or `opencl` (and optionally model
names). Raw numeric failure is an expected nonzero result for the eight
explicit deviation legs; use the separately approved task checker for them.

## Future release process

The user explicitly requires GitFlow for any release. Current HEAD is
`95027c3` on main, with uncommitted reviewed work and no develop/release branch
or GitFlow configuration. A future release must establish the integration
branch, land reviewed work there, cut `release/0.2.0`, validate the exact
release commit, merge into main/tag, and back-merge into develop. Accepted AMD limitations must be carried into the release notes. This packet performs no branch initialization or release.

## Final packet acceptance

Independent read-only review: **PASS**, see [review.md](review.md). The final
single acceptance command is `python3 specs/2026-10-24-release-gates/acceptance.py`;
its structured result is saved in `evidence/acceptance.json`. It verifies pinned
sources/artifacts, all 15 approved task checks, the seven/eight raw distinction,
no-skip suite evidence, formatting and the generated catalog. It does not turn
excluded AMD work into a pass or publish a release.
