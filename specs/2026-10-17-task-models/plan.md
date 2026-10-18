# Plan — Task-Model Coverage on NERT

## Working Convention

Use one isolated worktree per phase (`wip/nert-task-<n>`). Specification and
status commits may go to `main`; implementation merges only with complete
validation evidence. The kernel-single-source question (`Q-1`) must be settled
before a phase adds a new GPU row. CPU parity work can start without it.

## Phase 1 — ResNet-18 on CUDA (Complete)

Run the CI fixture on the CUDA device and compare it with ONNX Runtime. The
expected result was a passing gate without new kernels because `Conv`, `Gemm`,
`Relu`, and the pooling operators already had CUDA rows. The gate passed; if a
new gap appears on another CUDA model, record it as a separate work item in
node-count order.

## Phase 2 — DETR Gap Analysis

Once the DETR export and parity harness exist (`Q-2`), inventory every node and
operator, record which ones lack a GPU row, and turn the missing rows into
ordered work items. This phase produces the inventory and backlog; it does not
implement kernels.

## Phase 3 — Open-Vocabulary Parity

Start with CPU-versus-ONNX-Runtime parity for OWLv2 and Grounding DINO. OWLv2
is complete on CPU and CUDA. Grounding DINO now has a loadable, converted
export, but still needs a full reference gate. After CPU parity, run the same
dispatch-gap analysis used in Phase 2 and close the remaining device gaps.

## Phase 4 — Living Coverage Matrix

Maintain a short matrix next to this specification that crosses every model
with every device. Update it whenever a kernel row or a full-model gate lands.
A missing check must become a roadmap item with an owner; it must never remain
an invisible hole.

The first measured pass is [`coverage.md`](coverage.md), dated 2026-10-18. It
records the kernel-table baseline, the loadable models, the opset-blocked
exports, pending gates, the portable-device backlog, and the Neuriplo adapter
limitation.

## Validation

Per-phase evidence is recorded in
[`validation.md`](validation.md) as it is measured. A model is marked as
passing only when a full-model run against the reference exists; operator
coverage alone is not a gate.
