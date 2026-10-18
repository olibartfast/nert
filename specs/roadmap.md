# NERT Roadmap

> Status: living roadmap, created 2026-10-10 on extraction from Neuriplo
> (`olibartfast/nert@81b6b1f`). History lives in neuriplo packets
> `2026-09-17-native-engine-cpu-spine`, `2026-10-01-yolo26-native`, and
> `2026-10-10-native-tasks-op-coverage` (tiers 1–3 done 2026-10-10).

## Status Key

- **Complete** — landed with evidence.
- **Next** — the active packet.
- **Planned** — ordered, not yet specified.
- **Blocked** — needs an identified decision.

## Phase 0 — CPU Reference Spine (Complete)

Parser, IR, shape inference, arena planner, executor, device seam, and the
ResNet-18 op set (`Conv`, `Gemm`, `MatMul`, `Add`, `Relu`, `MaxPool`,
`ReduceMean`, `Reshape`). Proven by ResNet-18 parity (max abs diff
3.14713e-05, budget 1e-4). History: neuriplo
`specs/2026-09-17-native-engine-cpu-spine/`.

## Phase 1 — YOLO / Detection Coverage (Complete)

23 new ops incl. `Mul/Div/Sub/Sigmoid/Cast/Softmax/ReduceMax/Mod/Equal/Where/
Concat/Split/Unsqueeze/Expand/Transpose/Slice/Gather/GatherElements/Resize/
Flatten/Shape/ConstantOfShape/TopK`, constant folding, int64/bool
intermediates, confident-row parity on a real image (6 rows, min IoU
0.999999). History: neuriplo `specs/2026-10-01-yolo26-native/`.

## Phase 2 — YOLO Completion + Transformer Encoders (Complete)

Tier 1 (`ConvTranspose/Pow/Tile/AveragePool/Identity/Floor` + folding,
DOUBLE narrowing, duplicate value-info, Resize sizes) and Tier 2
(`Erf/LayerNormalization/Sqrt/Squeeze/Tanh/IsNaN` + [D-8]–[D-13]) with
model-zoo parity over YOLO families and ViT/ViTPose/Depth-Anything-V2/
VideoMAE/ViViT/TimeSformer, plus Tier 3 DETR-family groundwork
(`GridSample/ReduceSum/Range/Clip/Log/Pad/Abs/Neg/Exp/Cos/Sin/Einsum/And/Not/
Greater/Less`, int64+bool I/O [D-14], double accumulation [D-15], DETR
comparison [D-16], RT-DETR/D-FINE/DEIM/DEIMv2 end to end). Evidence: neuriplo
`specs/2026-10-10-native-tasks-op-coverage/validation.md`.

## Phase 3 — RAFT + Open Vocabulary (In progress)

Active packet: `specs/2026-10-10-raft-open-vocab/`.

- Tier 4: `InstanceNormalization` + RAFT end to end ([Q-3] static-per-load).
  **Complete (Phase A, 2026-10-10)** — kernel, shape rule, loader entry,
  hand-computed tests; RAFT parity `max_scaled_diff=6.10e-05` (atol 1e-3).
- Tier 5: `ArgMax/Elu/LessOrEqual/ReduceL2/CumSum/EyeLike/GreaterOrEqual/
  Max/Min/Or/ScatterND` + OWLv2 / Grounding DINO end to end. **Kernels
  complete 2026-10-16** (all 11: shape rule, loader entry, hand-computed +
  negative tests, format clean, 8/8 non-fixture suites green); OWLv2 /
  Grounding DINO parity + regression manifests are the open integration
  gate (Neuriplo-side runs).
- Exit: every gated export loads and matches ONNX Runtime per the packet's
  [V-3] (atol 1e-3 scaled, detection/DETR rows confident-only); ResNet
  fixture ctest stays green; no existing test weakened.

## Phase 4 — Hardening + Policy (Complete 2026-10-16)

[Q-1] decided: require opset-18 exports + conversion note (README).
[Q-3] decided: static per load ([D-8], evidenced by RAFT parity).
[Q-4] decided: ResNet-18 fixture only in CI; zoo parity in Neuriplo.
Packet: `specs/2026-10-16-phase4-hardening/`. No new ops; no `src/`/
`test/` behavior change.

## Phase 5 — Namespace Rename (Complete 2026-10-16)

[Q-5] decided: `include/nert/` + namespace `nert` + `snake_case` filenames
in one breaking 0.2.0 commit (`wip/nert-phase5`), with the matching Neuriplo
adapter change (includes + namespace) still to land consumer-side.
Packet: `specs/2026-10-16-phase5-rename/`. No behavior change.

## Phase N1 — Device Layers (In progress)

Same model + same plan on-device, matching the CPU oracle within tolerance,
device chosen once at load. Vendor decision (2026-10-16, maintainer):
hybrid, libraries-first — cuDNN/cuBLAS on NVIDIA, HIP via hipify (both
targets), Vulkan-then-OpenCL for the portable fallback. Slice log:
`specs/2026-10-16-tier5-to-N1-orchestration.md` (authoritative).

Landed (2026-10-16/17, RTX 3060): N1a scaffold + pilot (Gemm/Relu/Add),
N1b cuDNN Conv, N1e-1 elementwise (Mul/Sub/Div) + N1e-2 pools + N1e-3
activations, N1f runtime-bench dedup, N1g-1..5b YOLO set + GPU GATE
(YOLO26n fully dispatched, 6/6 rows), N1d-M1..M4 OpenCL (device + 8 ops),
N1c-M1..M3 HIP-on-NVIDIA (device + 5 ops, hipify-perl + GPU-agnostic bar), RF-DETR norm corridor on CUDA (Sqrt/Erf/ReduceSum/LayerNormalization, 126 gap nodes, oracle-matched), R6 + Tier-5 rows on CUDA (31 kernels: unary/logic/compare/fold/move/reduce/scatter families, oracle-matched; GridSample + Einsum excluded).
Dev environment: `.devcontainer` (26.08 stack); host stays toolkit-minimal.

Queued: portable-backend parity toward YOLO order (HIP pools next), N1e
slice C bias fusion + executor activation fusion, bench Slice 5 (fixtures
for the R6/Tier-5 set + Reshape on every table), Vulkan
shader-toolchain, AMD runner (rocBLAS/MIOpen + HIP-amd build).

Landed 2026-10-18: N1d-M5 OpenCL Reshape, N1c-M5 HIP Reshape, N1e slice A
cached cuDNN Conv state (5.9x per-call), slice B opt-in cuDNN Frontend
Conv row.

Measurement note (2026-10-16, `specs/2026-10-16-kernel-bench-harness/`,
branch `feature/bench-playground-layout`): the opt-in `bench/` harness now
exists — per-kernel CPU-oracle-vs-device harnesses with timings, starting
with `gemm` on CUDA + OpenCL C/C++ (RTX 3060: exact match, 5/5 seeds). This
is tooling only: the runtime stays stdlib-only and the device seam is still
CPU-only. N1 wires the device layer; it does not re-prove the harness.

## Specification Rule

Non-trivial work gets a dated `specs/YYYY-MM-DD-feature-name/` packet with
`requirements.md`, `plan.md`, `validation.md` — validation defined before
code, evidence recorded after. Small contained changes get a short spec note;
trivial fixes need no packet.
