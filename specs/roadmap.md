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

## Phase 3 — RAFT + Open Vocabulary (Next)

Active packet: `specs/2026-10-10-raft-open-vocab/`.

- Tier 4: `InstanceNormalization` + RAFT end to end ([Q-3] static-per-load).
- Tier 5: `ArgMax/Elu/LessOrEqual/ReduceL2/CumSum/EyeLike/GreaterOrEqual/
  Max/Min/Or/ScatterND` + OWLv2 / Grounding DINO end to end.
- Exit: every gated export loads and matches ONNX Runtime per the packet's
  [V-3] (atol 1e-3 scaled, detection/DETR rows confident-only); ResNet
  fixture ctest stays green; no existing test weakened.

## Phase 4 — Hardening + Policy (Planned)

Resolve [Q-1] (opset policy), [Q-4] (CI zoo subset), [Q-3] if still open;
close-out per neuriplo packet T-10 adapted to this repo (roadmap gate,
CHANGELOG, consumer note). No new ops unless the survey demands it.

## Phase 5 — Namespace Rename (Planned, Blocked on maintainer)

[Q-5]: `engine/` + namespace `engine` → `nert` in one breaking 0.2 release
with the matching Neuriplo adapter change. Not mixed into operator work.

## Phase N1 — CUDA Device Layer (Planned)

Same model + same plan on CUDA, matching the CPU oracle within tolerance,
device chosen once at load. Depends on the vendor-kernel decision (first-party
vs opt-in cuBLAS/cuDNN baseline) — recorded in Neuriplo, decision needed here
before any accelerated kernel.

## Specification Rule

Non-trivial work gets a dated `specs/YYYY-MM-DD-feature-name/` packet with
`requirements.md`, `plan.md`, `validation.md` — validation defined before
code, evidence recorded after. Small contained changes get a short spec note;
trivial fixes need no packet.
