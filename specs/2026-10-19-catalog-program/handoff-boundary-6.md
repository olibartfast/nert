# Handoff — NERT catalog coverage, boundary 6 of N

Date: 2026-10-20. Written by the closing orchestrator session, which ran the
**RAFT-small gate**. Read this file, the ledger, the inventory and the coverage
matrix; nothing else is needed to continue.

Placeholders: `$NERT_REPO` = NERT root; `$LOCAL_ARTIFACTS` = local artifacts
root (`tritonic/`); `$HERMES_CACHE` = pre-slim archive root; `${HOME}` = home.

## Where the program stands

- Repo: `$NERT_REPO`, branch `main`, commit `0a54c26`, pushed to `origin/main`,
  tree clean. **Scoreboard: PASS 22 / Open 1 / Missing export 7 / Out of scope
  2** (22 + 1 + 7 = 30 in-scope of 32). Open entry: Grounding DINO (OpenCL
  backend GAP; GAP-vs-fix decision with maintainer — do not implement).
- **0.2.0 is not release-ready.** No release cut, no tag.
- Kernel tables unchanged: CPU 72, CUDA 72, OpenCL 68, HIP 9 (unsupported).

## What closed this boundary

**RAFT-small (entry 31)** — full CPU/CUDA/OpenCL gate on the existing opset-18
artifact (`/tmp/opencode/exports/raftsmall/prov/raft_small-18.onnx`, sha256
`8be6b770…`, 1,630 nodes on disk / **1,115 dispatched** equally). Frozen
fixture frame1 `95c3b7f0…` / frame2 `870533ce…`; ORT 1.27.0 CPU ref
`a622b211…`. `--tol 1e-3` scaled, over_tol **0** all legs; max_abs CPU/CUDA
6.10e-05, OpenCL 6.20e-05 (393,216 elements). Gate reviewer
**PASS-WITH-FINDINGS** (non-blocking); integration reviewer first FAIL (stale
ledger Status + README), corrected and re-reviewed **PASS**. Evidence:
`/tmp/opencode/gates/raft_small/raft_small_gate_evidence.md`.

## Boundary-7 order (maintainer-set, unchanged)

1. **VideoMAE / ViViT / TimeSformer** — export opset 18, probe, gate each; one
   shared frozen static 5D fixture. NEXT.
2. **RF-DETR det/seg/keypoint other sizes** — fix `rfdetr`/transformers in an
   isolated venv under `/tmp/opencode/`; weights download non-hermetic, record
   in provenance.
3. **EdgeCrafter det/seg/pose** — last, heaviest; `orig_target_sizes` int64
   payload.

Each model gets its own boundary: gate → reviewer → integration → commit →
fresh orchestrator reset. One packet per family; at most two concurrent packets
with disjoint writable paths; GPU legs serialized.

## Standing rules (carry forward)

- Gate harness `/tmp/opencode/nert_gate.cpp`; `gate_compare.py` always exits 0
  — verdict from over-tolerance counts. Acceptance asserts captured `.exit`
  files, never self-attested strings. `Constant` folds in the loader read loop.
- Downloads/clones/isolated venvs under `/tmp/opencode/` authorized. Never
  modify the Neuriplo repo, global packages, or repo dependencies. Missing
  checkpoint/credential → stop `BLOCKED — exact missing input`.
- Specs stay slim: concise evidence and counts only.

## Session reset

```
opencode api session.create -d '{"title":"NERT catalog program — boundary 7"}'
opencode run --agent orchestrator --session <id> --auto "<handoff prompt>" &
```

New session reads only this handoff, the ledger, the inventory and coverage.
Record the new session id in the ledger's "Sessions and resets" table.
