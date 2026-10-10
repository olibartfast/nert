# Validation — RAFT + Open-Vocabulary Operator Coverage on NERT

Written before implementation. Nothing is checked unless it ran.

## Immutable Acceptance (planner-owned; workers run it once, as final action)

```bash
cmake -S . -B build
cmake --build build
python3 tools/provision_resnet18.py fixtures   # once; needs torch + torchvision 0.27.0
NERT_FIXTURE=fixtures/resnet18.onnx ctest --test-dir build --output-on-failure
scripts/format.sh --check
git status --short
```

Expected: configure + build clean; all `engine_*` suites pass; format clean;
`git status` shows only intended files.

## Requirement Checks

- [x] [V-1] → [R-1]/[R-2]: every new operator (1 tier-4 + 11 tier-5) has a
  hand-computed unit case in `test/KernelsTest.cpp` (computed by hand, not
  captured), off-default attributes included, plus one negative load/shape
  case naming the op. Command: `ctest --test-dir build -R Kernels`.
  **Phase A only:** tier-4 `InstanceNormalization` (incl. off-default
  `epsilon`, per-channel scale/bias, negative cases) is covered; the 11
  tier-5 operators are open — Phase B.
- [x] [V-2] → [R-3]/[R-4]/[R-6]: ResNet-18 fixture ctest green (immutable
  acceptance above) — no regression from new kernels.
- [x] [V-3] → [R-3]: RAFT parity evidence from Neuriplo
  (`nert_model_zoo_parity_test` on `tier4_raft.manifest`, real pair,
  `parity`, atol 1e-3 scaled): record export, opset, max scaled diff, date.
  Missing fixture/input fails, never skips.
- [ ] [V-4] → [R-4]: OWLv2 + Grounding DINO rows evidence from Neuriplo
  (same harness): record export, opset, mode, max scaled diff / confident
  rows, date. **Open — Phase B.**
- [ ] [V-5] → [R-5]/[R-6]: regression evidence — YOLO + transformer + DETR
  manifests green on the same Neuriplo pin; `git diff --stat` reviewed for
  scope (new kernels + loader/shape/test entries only, no executor design
  changes, no existing test weakened). **Open — Phase B.**
- [x] [V-6] → constraints: `scripts/format.sh --check` clean; no new runtime
  dependency (`grep` link closure / CMakeLists review); opset-18-only loader
  unchanged (no silent opset widening).

## Evidence Log

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | `NERT_FIXTURE=fixtures/resnet18.onnx ctest --test-dir build -R Kernels` | pass | 2026-10-10 | tier-4 `InstanceNormalization`: hand-computed cases incl. off-default `epsilon`, per-channel scale/B, and negative load/shape cases. Tier-5 operators not yet implemented (Phase B). |
| V-2 | `NERT_FIXTURE=... ctest --test-dir build` | 12/12 pass | 2026-10-10 | ResNet-18 fixture; no regression from the new kernel. |
| V-3 | `NEURIPLO_MODEL_ZOO_MANIFEST=~/.cache/neuriplo-op-inventory/tier4_raft.manifest build-ne/backends/nert/test/nert_model_zoo_parity_test` | pass | 2026-10-10 | torchvision RAFT-large, opset 16→18, `parity`, real `bus.jpg` pair 256×320 in [-1,1] (+4 px shift); `max_abs_diff=6.94e-05`, `max_scaled_diff=6.10e-05`, atol 1e-3. Neuriplo build: NERT + ONNX_RUNTIME (ORT 1.28.0), `-DFETCHCONTENT_SOURCE_DIR_NERT=../nert-w-p4a`. |
| V-6 | `scripts/format.sh --check` | pass | 2026-10-10 | 58 files; no new runtime dependency (stdlib-only kernel). |

## Deviations

_Record here any criterion not fully met, why, and how it is tracked._

## Definition of Done (integration)

- [ ] Immutable acceptance green with evidence + dates
- [ ] [V-1]–[V-6] evidenced; deviations documented
- [ ] Spec, code, roadmap, CHANGELOG agree — merged as one coherent change
