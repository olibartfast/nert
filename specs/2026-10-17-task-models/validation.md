# Validation — Other task models on NERT

Defined before implementation. Bars: classification matches top-1 with
score diff at or under 1e-4 on CPU; GPU deviations follow the R-6
precedent (measured, detection-identical, maintainer-noted).

## Checks

- [V-1] ResNet-18 on CUDA: top-1 matches ORT, score diff recorded.
- [V-2] DETR: export inventoried (node/op list on file), GPU gaps
  enumerated as slice entries.
- [V-3] OWLv2 / Grounding-DINO: CPU parity numbers recorded; GPU gaps
  enumerated as slice entries.
- [V-4] Coverage matrix exists and is current through these phases.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| — | — | pending | — | — |
