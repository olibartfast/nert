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

- [ ] [V-1] → [R-1]/[R-2]: every new operator (1 tier-4 + 11 tier-5) has a
  hand-computed unit case in `test/KernelsTest.cpp` (computed by hand, not
  captured), off-default attributes included, plus one negative load/shape
  case naming the op. Command: `ctest --test-dir build -R Kernels`.
- [ ] [V-2] → [R-3]/[R-4]/[R-6]: ResNet-18 fixture ctest green (immutable
  acceptance above) — no regression from new kernels.
- [ ] [V-3] → [R-3]: RAFT parity evidence from Neuriplo
  (`nert_model_zoo_parity_test` on `tier4_raft.manifest`, real pair,
  `parity`, atol 1e-3 scaled): record export, opset, max scaled diff, date.
  Missing fixture/input fails, never skips.
- [ ] [V-4] → [R-4]: OWLv2 + Grounding DINO rows evidence from Neuriplo
  (same harness): record export, opset, mode, max scaled diff / confident
  rows, date.
- [ ] [V-5] → [R-5]/[R-6]: regression evidence — YOLO + transformer + DETR
  manifests green on the same Neuriplo pin; `git diff --stat` reviewed for
  scope (new kernels + loader/shape/test entries only, no executor design
  changes, no existing test weakened).
- [ ] [V-6] → constraints: `scripts/format.sh --check` clean; no new runtime
  dependency (`grep` link closure / CMakeLists review); opset-18-only loader
  unchanged (no silent opset widening).

## Evidence Log

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| — | — | — | — | _to be recorded on execution_ |

## Deviations

_Record here any criterion not fully met, why, and how it is tracked._

## Definition of Done (integration)

- [ ] Immutable acceptance green with evidence + dates
- [ ] [V-1]–[V-6] evidenced; deviations documented
- [ ] Spec, code, roadmap, CHANGELOG agree — merged as one coherent change
