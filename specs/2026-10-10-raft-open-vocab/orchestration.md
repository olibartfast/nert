# Orchestration — RAFT + Open-Vocabulary on NERT

Workflow: `orchestrate-ai-coding-workflows`. SDD packet
(`requirements.md`, `plan.md`, `validation.md`) is committed before any
worker dispatch; spec files stay orchestrator-owned.

## Roles and routing

| Role | Who | Tier | Owns |
| --- | --- | --- | --- |
| Planner / orchestrator | main session | strongest | packets, acceptance runs, integration, commits, this file |
| Implementer | subagent, one per phase, fresh context | mid | the packet's writable paths only |
| Reviewer (optional) | read-only subagent | strongest | review report, never a diff |

Defects go back to a fresh implementer in a corrected packet; the planner
does not patch worker output. Model-zoo parity runs in Neuriplo (minutes per
export) are planner-run after integration, never worker acceptance — the
worker's acceptance is the immutable ResNet fixture ctest only.

## Boundaries: enforced vs advisory

- **Enforced:** each implementer runs in its own git worktree
  (`../nert-w-<phase>`, branch `wip/nert-<phase>`, from the packet's starting
  revision); reviewer has no write tools; planner is the only role that
  commits.
- **Advisory (review obligation):** writable-path lists and the never-commit
  rule — the planner checks `git status` / `git diff --stat` before
  integrating and rejects out-of-scope edits.
- **Resource limits:** builds `--parallel 2` (maintainer RAM constraint);
  one implementer at a time.

## Acceptance (planner-owned; workers run it once, final action)

```bash
cmake -S . -B build && cmake --build build \
  && NERT_FIXTURE=fixtures/resnet18.onnx ctest --test-dir build --output-on-failure \
  && scripts/format.sh --check && git status --short
```

Integration gate (planner, in Neuriplo after merging a phase): NERT + ONNX_RUNTIME
build, `NertInferTest`, ResNet-18 + YOLO26 parity, then
`nert_model_zoo_parity_test` on the touched families plus YOLO/tier-2/DETR
regression.

## Run ledger

| Attempt | Packet | Implementer | Acceptance (first pass) | Integration gate | Notes |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | _packet committed; no attempts yet_ |
