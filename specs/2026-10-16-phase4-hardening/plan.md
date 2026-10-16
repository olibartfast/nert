# Plan — Phase 4 Hardening + Policy

Single phase (docs only), then close-out.

## P1 — Policy records + conversion note

- `README.md`: opset-18 conversion note with the exact command
  (`onnx.version_converter.convert_version` to opset 18 before export).
- `specs/tech-stack.md`: [Q-1] → decided (require 18 + note); [Q-3] →
  decided (static per load, evidenced by RAFT parity); [Q-4] → decided
  (ResNet fixture only in CI; zoo parity in Neuriplo).
- `specs/roadmap.md`: Phase 4 → Complete with decision dates.
- `specs/2026-10-16-phase4-hardening/validation.md`: evidence after execution.

## P2 — Close-out

- CHANGELOG already carries Tier-5 (see `ff3eba1`); verify entry renders.
- `scripts/format.sh --check` clean (docs edits must not touch sources);
  `git status`/`diff --stat` scope review.
