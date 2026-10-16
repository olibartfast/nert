# Validation — Phase 4 Hardening + Policy

Defined before implementation. Evidence after execution.

## Checks

- [V-1] → [R-1]: README names opset-18-only loading + the conversion
  command; loader code untouched (`git diff --stat` shows no `src/`).
- [V-2] → [R-2]/[R-3]: tech-stack [Q-1]/[Q-3]/[Q-4] marked decided with
  dates; CI workflow unchanged.
- [V-3] → [R-4]: roadmap Phase 4 → Complete; CHANGELOG Tier-5 entry
  present; `scripts/format.sh --check` clean; `git status` shows only
  docs/specs.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| — | — | — | — | _to fill_ |
| V-1 | README opset note + `git diff --stat` scope | pass | 2026-10-16 | conversion note added; no `src/`/`include/`/`test/`/`bench/` changes. |
| V-2 | tech-stack Q-1/Q-3/Q-4 decided | pass | 2026-10-16 | CI workflow untouched (already ResNet-only). |
| V-3 | roadmap Phase 4 Complete + format check | pass | 2026-10-16 | `scripts/format.sh --check` clean (docs-only). |
