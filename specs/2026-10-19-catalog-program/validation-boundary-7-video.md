# Validation — boundary 7 video (VideoMAE, ViViT, TimeSformer)

Date: 2026-10-20. Scope: entry 29 integration only. Three frozen opset-18
exports, one shared fixture family, nine full-model gate legs. No engine or
kernel change.

## Preconditions

| packet | check | result |
|---|---|---|
| P1 VideoMAE | `/tmp/opencode/gate/video-videomae/gate_result.json` | `over_tol == 0` on cpu/cuda/opencl; verdict PASS; REVIEW-PASS |
| P2 ViViT | `/tmp/opencode/gate/video-vivit/gate_result.json` | `over_tol == 0` on cpu/cuda/opencl; verdict PASS; REVIEW-PASS |
| P3 TimeSformer | `/tmp/opencode/gate/video-timesformer/gate_result.json` | `over_tol == 0` on cpu/cuda/opencl; verdict PASS; REVIEW-PASS |
| Repo before branching | `git status --short --branch` | clean, `main...origin/main` at `47bfe8f` |

## Frozen artifacts

| model | export sha256 | nodes disk / dispatched | input | fixture sha256 | reference sha256 | opset |
| --- | --- | --- | --- | --- | --- | ---: |
| VideoMAE | `14fe0ace…05cb586` | 521 / 407 | `[1,16,3,224,224]` f32 | `537c5e90…2e0fc` | `20efdf6c…95e2` | 18 |
| ViViT | `24d5690a…42d39` (raw `6300d277…550b38d1`) | 1,152 / 797 (`If`=0, `Identity`=3) | `[1,32,3,224,224]` f32 | `78157408…67536` | `9fd575da…03f4b` | 18 |
| TimeSformer | `f0256c37…035a` | 1,289 / 904 | `[1,8,3,224,224]` f32 | `2d10ec9b…4eb564` | `150a1393…1e078` | 18 |

ViViT raw-vs-folded logits are bit-identical (`max_abs` 0.0); the raw export
is preserved. Output is `logits [1,400]` float32 for all three.

## Gate results

`--tol 1e-3`, scaled threshold `tol * (1 + |ref|)`. Nine legs, all exit 0.

| model | backend | over_tol | max_abs | max_ratio |
| --- | --- | ---: | ---: | ---: |
| VideoMAE | cpu | 0/400 | 1.3113021850585938e-06 | 0.0009440241920040157 |
| VideoMAE | cuda | 0/400 | 1.2516975402832031e-06 | 0.0008659470031916535 |
| VideoMAE | opencl | 0/400 | 1.9669532775878906e-06 | 0.0009925095604235309 |
| ViViT | cpu | 0/400 | 7.3909759521484375e-06 | 0.004278399711686194 |
| ViViT | cuda | 0/400 | 8.344650268554688e-06 | 0.004494117344208187 |
| ViViT | opencl | 0/400 | 3.8623809814453125e-05 | 0.031353039818606473 |
| TimeSformer | cpu | 0/400 | 3.337860107421875e-06 | 0.0018960872446111409 |
| TimeSformer | cuda | 0/400 | 3.814697265625e-06 | 0.002557859409865797 |
| TimeSformer | opencl | 0/400 | 6.67572021484375e-06 | 0.003530061276282632 |

References: ORT 1.27.0 `CPUExecutionProvider`, frozen fixture seed `20261020`
(canonical 32-frame prefix; F=16/32/8). Reference argmax: VideoMAE 143,
ViViT 382, TimeSformer 304.

## Reviews

| model | verdict | reviewer session |
| --- | --- | --- |
| VideoMAE | REVIEW-PASS | `ses_ea51f4c3cffeiJ8JfMwkHheTHx` |
| ViViT | REVIEW-PASS | `ses_ea51f4c27ffen77q2lg0kFKGlS` |
| TimeSformer | REVIEW-PASS | `ses_ea4fdff42ffeNIjiWGQPc5mTyJ` |

## Branch and integration

- `git branch -a | grep -i develop` returned nothing (exit 1). No `develop`
  branch exists locally or on any remote — expected absent and recorded.
- Integration branch: `wip/nert-b7-video`, created from current `origin/main`
  (`47bfe8f`), never from `develop`. Single commit; no push, no merge to
  `main`, no tag, no release.
- Orchestrator session: `ses_ea57088a6ffeiju9apmez1FxlW`.
- Acceptance: `bash /tmp/opencode/specs/video-integration/acceptance.sh`
  (run once after the integration commit); evidence in the same directory.

## Harness provenance caveat

The prebuilt `nert_gate_{cpu,cuda,opencl}` binaries predate HEAD `47bfe8f`
(`bench: typed fixtures for R6/Tier-5 rows`, a bench-only commit touching
`bench/` and the bench-dedup validation). TimeSformer's dispatched op set
contains no `Resize`, so its numerics are unaffected; CPU/CUDA/OpenCL agree
with each other and the ORT reference (0/400 over tolerance on every leg).
