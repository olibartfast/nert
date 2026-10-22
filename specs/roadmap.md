# NERT Roadmap

> Status: living roadmap. Created on 2026-10-10 when NERT moved out of
> Neuriplo (`olibartfast/nert@f9981db`). Detailed history remains in the
> Neuriplo packets `2026-09-17-native-engine-cpu-spine`,
> `2026-10-01-yolo26-native`, and `2026-10-10-native-tasks-op-coverage`.

## How to Read the Status

- **Complete** — implemented and backed by recorded evidence.
- **In progress** — active work with a current specification.
- **Planned** — ordered, but not yet specified.
- **Blocked** — waiting on a decision, export, or external toolchain.

## Phase 0 — CPU Reference Spine (Complete)

The first phase delivered the interpreter itself: ONNX parsing, the graph IR,
static shape inference, arena planning, the executor, the device seam, and the
ResNet-18 operator set (`Conv`, `Gemm`, `MatMul`, `Add`, `Relu`, `MaxPool`,
`ReduceMean`, and `Reshape`). ResNet-18 parity against ONNX Runtime recorded a
maximum absolute difference of `3.14713e-05` against a `1e-4` budget.

History: Neuriplo specification packet
`specs/2026-09-17-native-engine-cpu-spine/`.

## Phase 1 — YOLO and Detection Coverage (Complete)

Twenty-three operators were added for YOLO detection, including `Mul`, `Div`,
`Sub`, `Sigmoid`, `Cast`, `Softmax`, `ReduceMax`, `Mod`, `Equal`, `Where`,
`Concat`, `Split`, `Unsqueeze`, `Expand`, `Transpose`, `Slice`, `Gather`,
`GatherElements`, `Resize`, `Flatten`, `Shape`, `ConstantOfShape`, and `TopK`.
The phase also introduced constant folding, int64 and bool intermediates, and
confident-row parity on a real image: six rows matched with a minimum IoU of
`0.999999`.

History: Neuriplo specification packet
`specs/2026-10-01-yolo26-native/`.

## Phase 2 — YOLO Completion and Transformer Encoders (Complete)

This phase completed the YOLO families and added the transformer encoder
operators. Tier 1 covered `ConvTranspose`, `Pow`, `Tile`, `AveragePool`,
`Identity`, and `Floor`, along with folding and dtype narrowing fixes. Tier 2
added `Erf`, `LayerNormalization`, `Sqrt`, `Squeeze`, `Tanh`, and `IsNaN` and
proved parity for ViT, ViTPose, Depth-Anything-V2, VideoMAE, ViViT, and
TimeSformer.

Tier 3 supplied the DETR groundwork: `GridSample`, `ReduceSum`, `Range`,
`Clip`, `Log`, `Pad`, `Abs`, `Neg`, `Exp`, `Cos`, `Sin`, `Einsum`, `And`,
`Not`, `Greater`, and `Less`, plus int64 and bool graph I/O, double
accumulation for reductions, DETR comparison rules, and end-to-end runs for
RT-DETR, D-FINE, DEIM, and DEIMv2.

Evidence: Neuriplo specification
`specs/2026-10-10-native-tasks-op-coverage/validation.md`. The related
decision records are `D-8` through `D-16`, including `D-13` for the
Resize-size policy, `D-14` for int64 and bool I/O, and `D-15` for double
accumulation.

## Phase 3 — RAFT and Open Vocabulary (In Progress)

Active specification: `specs/2026-10-10-raft-open-vocab/`.

- **RAFT (Tier 4).** The `InstanceNormalization` kernel, shape rule, loader
  entry, and tests are complete. RAFT parity recorded
  `max_scaled_diff=6.10e-05` against an `atol` of `1e-3`. The real
  torchvision RAFT-large export is not present in this repository; the parity
  number came from Neuriplo's manifest harness.
- **Open-vocabulary operators (Tier 5).** All eleven kernels are implemented:
  `ArgMax`, `Elu`, `LessOrEqual`, `ReduceL2`, `CumSum`, `EyeLike`,
  `GreaterOrEqual`, `Max`, `Min`, `Or`, and `ScatterND`. OWLv2 is complete on
  CPU and CUDA. Grounding DINO now has a converted, loadable export but still
  needs a full reference gate.
- **OWLv2 bring-up.** Real-model dispatch exposed and closed several gaps:
  int32 `Cast` emulation on int64 lanes, Einsum ellipsis handling, bool
  movement (`Flatten`, `Gather`, `Concat`, `Slice`, `Transpose`, `Unsqueeze`),
  empty `ConstantOfShape`, and CUDA `Einsum`. CPU results match ONNX Runtime:
  logits are bit-identical (top-1 `3600/3600`) and box differences peak at
  `1.3e-04`. CUDA completes the full 1219-node dispatch in about seven seconds
  with bit-identical logits and box differences peaking at `1.9e-04`; no row
  exceeds the tolerance. Evidence: validation `V-3` and `V-3b` in the
  task-model specification.

The exit condition is that every gated export loads and matches ONNX Runtime
under the specification's validation rules, the ResNet fixture stays passing,
and no existing test is weakened. Current model-level status is tracked in
`specs/2026-10-17-task-models/coverage.md`.

## Phase 4 — Hardening and Policy (Complete, 2026-10-16)

This phase settled the policy questions that affect every later feature:

- `Q-1`: require opset-18 exports and document the conversion step.
- `Q-3`: resolve RAFT's symbolic spatial dimensions statically per load.
- `Q-4`: keep CI to the ResNet-18 fixture; run model-zoo parity in Neuriplo.

The specification is `specs/2026-10-16-phase4-hardening/`. No operators or
runtime behavior changed.

## Phase 5 — Namespace Rename (Complete, 2026-10-16)

The breaking 0.2.0 rename moved public headers to `include/nert/`, the
namespace to `nert`, and C++ filenames to lowercase `snake_case` (decision
`Q-5`). The Neuriplo adapter still needs the matching include and namespace
update on its side.

Specification: `specs/2026-10-16-phase5-rename/`. No behavior changed.

## Phase N1 — Device Layers (In Progress)

The device-layer goal is simple to state: the same model and the same plan run
on a device and match the CPU oracle within tolerance, with the device chosen
once at load. The vendor strategy is hybrid and libraries-first: cuDNN and
cuBLAS on NVIDIA, HIP through hipify where the toolchain exists, and a portable
fallback that starts with OpenCL.

The authoritative work log is
`specs/2026-10-16-tier5-to-N1-orchestration.md`.

What is complete:

- The CUDA device layer, including the YOLO operator set and the full-model
  YOLO26n gate (six confident rows, minimum IoU `0.9998039`).
- cuDNN `Conv`, cuDNN backward-data `ConvTranspose`, and the RF-DETR norm
  corridor (`Sqrt`, `Erf`, `ReduceSum`, `LayerNormalization`).
- The R6 and Tier-5 CUDA kernels, including `GridSample`,
  `InstanceNormalization`, and the remaining reduction, comparison, and
  scatter operators. The CUDA table now covers every CPU operator, including
  `Einsum`, which received its CUDA row during the OWLv2 bring-up.
- OpenCL device support and the movement, pool, reshape, `Conv`, `Flatten`,
  and `Gemm` rows. The OpenCL table is still being expanded; the live
  model-by-device matrix is in `specs/2026-10-17-task-models/coverage.md`.
- HIP device support for the original nine operators. HIP parity for the
  newer rows is blocked on a host with `hipify-perl`, ROCm, and AMD hardware.

The development environment is the `.devcontainer` 26.08 stack; the host stays
toolkit-minimal.

Still queued: the remaining OpenCL transformer and reduction rows, HIP parity
where hardware permits, bias fusion and executor activation fusion, additional
bench fixtures, a Vulkan shader toolchain, and an AMD runner built on rocBLAS
and MIOpen.

The opt-in `bench/` harness exists under
`specs/2026-10-16-kernel-bench-harness/`. It provides per-kernel CPU-oracle
versus device comparisons with timings, starting with `gemm` on CUDA and
OpenCL C/C++. It is measurement tooling only; the runtime remains
standard-library-only.

## Opset Range and Versioned Loading (implemented: 11+)

Two linked packets replace the exact-opset-18 gate with a verified policy.

- **≤18 path** — `specs/2026-10-21-opset-range/`. Phase A accepts declared
  opsets 13–18 by canonicalizing old node forms to the opset-18 IR (the only
  13→18 rewrite is the `axes` attribute-to-input move on `ReduceMean`,
  `ReduceL2`, and `ReduceMax`); Phase B extends the floor to 11: legacy
  `Softmax` coercion, `ReduceSum` / `Squeeze` / `Unsqueeze` axes, and
  `Split` split. Opset 10 and below stay deferred with the conversion note.
- **>18 path** — `specs/2026-10-21-opset-newer-than-18/`. Adopts OpenCV 5's
  “declared opset is data” acceptance model, but replaces its silent
  fallbacks with a committed generated per-operator compatibility table and
  explicit guards (`Cast.saturate`, `Cast.round_mode`, `Pad.wrap`,
  `AveragePool.dilations`, `Range.stash_type`, `GridSample` 20+). It uses
  the ≤18 packet's opset plumbing and has no numeric ceiling.

Status: supported declarations are **11 and above**. Phase A validated
2026-10-21; Phase B and the newer-than-18 path validated 2026-10-22.
`specs/tech-stack.md` `Q-1` records the revised policy.

Follow-up capability slices (rejected explicitly today): `Cast.saturate`,
`Cast.round_mode`, `Pad.wrap`, `AveragePool.dilations`, `Range.stash_type`,
and `GridSample` opset-20 semantics.

### Version-aware opsets (Phases A–B validated 2026-10-22; Phases C–E pending)

`specs/2026-10-22-version-aware-opsets/` replaces the global opset-18
canonical target with per-node effective schema metadata and version-aware
shape/kernel resolution. Phase A adds `Graph.declared_opset` and
`Node.schema_version` and extends the generated catalog to the full schema
history while keeping the normalizer behavior-neutral. Phase B keeps the
pre-18 reduce `axes` attribute in the graph and resolves it in shape
inference plus the CPU/CUDA/OpenCL reduce rows; the reduce rewrite is gone.
Later phases remove the remaining axes/split rewrites, implement legacy
`Softmax` directly, and delete the normalizer. This supersedes decisions
`D-1` and `D-4` of the opset-range packet.

## Tooling — Memory Footprint Report (Planned)

The planned model fit/sizing report adds `MemoryFootprint` to `Model` and an
opt-in `nert_mem_report` command-line tool. It accounts for NERT-managed bytes
by category — arena, initializers, inputs and outputs, and device-resident
copies — together with the planned arena peak.

Specification: `specs/2026-10-18-memory-footprint/`. User guide:
`docs/memory-footprint-profiling.md` (currently marked planned). The report
excludes measured resident set size (RSS), device usage, and kernel
workspaces. Implementation has not started.

## Specification Rule

Non-trivial work gets a dated specification under
`specs/YYYY-MM-DD-feature-name/` with `requirements.md`, `plan.md`, and
`validation.md`. Validation is defined before implementation, and evidence is
recorded after the checks run. Small, contained changes may use a short spec
note; trivial fixes need no specification.
