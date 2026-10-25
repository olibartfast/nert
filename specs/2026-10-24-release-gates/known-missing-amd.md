# Accepted known missing coverage — 0.2.0

Maintainer decision, 2026-10-24: record full-catalog native AMD/ROCm model
validation and persistent AMD GPU CI as **known missing coverage**, rather
than blocking 0.2.0 on those checks. This supersedes the mandatory AMD release
requirement adopted on 2026-10-23.

- Full-catalog native AMD parity remains UNRUN.
- Persistent AMD CI is not registered/validated; skipped jobs are not PASS.
- Earlier MI300X build/device and ResNet/YOLOv8n/Depth-Anything checks remain
  artifact-specific historical evidence and predate the latest Sign/Split fixes.
- Current HIP/NVIDIA checks do not substitute for native AMD execution.

Carry these limitations into the published release notes. Do not mark unrun
validation checkboxes complete or claim complete AMD support. Existing signed
review and acceptance artifacts retain the release-policy state at execution;
this later policy decision changes release eligibility, not their test results.

The local gate packet remains accepted. Changes must still be committed and
released through GitFlow; this decision does not publish or authorize a tag.
