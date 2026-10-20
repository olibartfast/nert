# Handoff — NERT catalog coverage, boundary 4 of N

Date: 2026-10-20. Written by the closing orchestrator session, which ran the
**Wave-A** tranche of the export-first provisioning batch (ViTPose, plus the
Depth-Anything-V2 and RAFT-small provisions). Read this file, the ledger, the
inventory and the coverage matrix; nothing else is needed to continue.

**Placeholders** (the program's convention; no local home paths are recorded
here). `$NERT_REPO` = the NERT repository root; `$LOCAL_ARTIFACTS` = the local
artifacts root that holds `tritonic/`; `$HERMES_CACHE` = the cache root that
holds `cache/scratch/nert-specs-slim-2026-10-19/` with the pre-slim coverage,
inventory and ledger; `${HOME}` = the user home.

## Where the program stands

- Repo: `$NERT_REPO`, branch `main`. **Scoreboard: PASS 20 / Open 1 /
  Missing export 9 / Out of scope 2** (20/1/9/2; 20 + 1 + 9 = 30 in-scope
  entries of 32). The one open entry is Grounding DINO, whose OpenCL cell is a
  **backend gap**.
- **0.2.0 is not release-ready**: the gate needs zero open entries *and* zero
  missing exports, and nine entries still have no gated export. **No release
  was cut, no tag was made, and no release is recommended.**
- Kernel tables **unchanged**: CPU 72, CUDA 72, OpenCL 68, HIP 9 (unsupported
  for 0.2.0). **No kernel row was added this boundary and no engine source
  file changed.**
- The coverage matrix and the census carry the full 20/1/9/2 scoreboard; the
  README carries the 20 passing, 1 open and 9 missing-export counts; the
  CHANGELOG carries the 19 → 20 and 10 → 9 moves and "No release was cut".

## What closed this boundary

1. **ViTPose-base-simple (entry 26)** is exported at **opset 18** and gated on
   CPU, CUDA and OpenCL. HF `usyd-community/vitpose-base-simple` (commit
   `a93ac0c6…`); 536 nodes on disk / **422 dispatched** on every device (the
   114 `Constant` nodes are folded at load and never dispatched);
   `pixel_values [1,3,256,192]` float32 → `heatmaps [1,17,64,48]` float32;
   export sha256 `b72b99e5…`. Every leg exited 0 with over_tol **0** on all
   three devices at `--tol 1e-3` scaled. Host-side keypoint argmax agreed
   17/17 on every device (outside the ONNX gate).
2. **Two exports were provisioned and provenance-reviewed but NOT probed or
   gated**, and they do not count as PASS. **They are the first two packets of
   boundary 5, and they are GATE packets, not export packets — do not
   re-export them.**
   - **Depth-Anything-V2-Small** — opset 18, 553 nodes on disk, export sha256
     `497acfff…`; provenance at `/tmp/opencode/exports/depthv2/prov/provenance.md`
     (reviewed, PASS-WITH-FINDINGS, all findings corrected in place). Its
     fixture is frozen: `input_pixel_values.bin` sha256 `6a50426d…`, tensor
     `pixel_values [1,3,518,518]` → `predicted_depth [1,518,518]`.
   - **RAFT-small** — opset 18 by conversion from an opset-16 export, 1,630
     nodes on disk, export sha256 `8be6b770…`; provenance at
     `/tmp/opencode/exports/raftsmall/prov/provenance.md` (reviewed,
     PASS-WITH-FINDINGS, all findings corrected in place). Its **image-pair**
     fixture is frozen: `frame1.bin` sha256 `95c3b7f0…`, `frame2.bin`
     sha256 `870533ce…`, both `input1`/`input2` `[1,3,384,512]` →
     `flow_prediction [1,2,384,512]`. The conversion caveat and an
     ORT-vs-ORT pre/post-conversion diff are already recorded.
   - Both still need their **PROBE** (op census + per-device missing-row diff +
     a CPU load proof) and then their full **CPU/CUDA/OpenCL GATE**.
3. **The Grounding DINO OpenCL remedy was ASSESSED, not implemented** (see
   below). The OpenCL cell stays `GAP`; Grounding DINO remains the one open
   entry.

## Reusable method and exact template paths

Five steps per group, unchanged: **PROVISION → PROBE → GATE → REVIEW →
INTEGRATE**, with a fresh read-only reviewer after every implementer packet.

- Export templates (**copy these four**):
  `/tmp/opencode/exports/resnet50/verify_export.py`,
  `/tmp/opencode/exports/resnet50/make_census.py`,
  `/tmp/opencode/exports/resnet50/probe_accept.sh`,
  `/tmp/opencode/exports/resnet50/provenance.md`.
- Gate harness: `/tmp/opencode/nert_gate.cpp`; comparison tool
  `/tmp/opencode/gate_compare.py` (report-only — its exit code is always 0).
- Build scripts: `/tmp/opencode/build_gate_*.sh` (the `relink.stamp` records
  the last relink: `/tmp/opencode/relink.stamp`).
- Export artifacts live under `/tmp/opencode/exports/<group>/`, references
  and fixtures under `/tmp/opencode/catalog_refs/<model>/`, evidence under
  `/tmp/opencode/gates/`.

## Environment facts (measured)

- python **3.12.3**; torch **2.12.0+cu130**; `transformers` **4.35.0**
  system-wide and **4.57.1** inside an export venv; onnx **1.22.0**;
  onnxruntime **1.27.0**; ultralytics **8.4.113**; **network UP**.
- GPU: **RTX 3060 Laptop, 6144 MiB**. It is a single-owner resource: only one
  GPU leg at a time.
- Three build archives: `build/` (CPU), `build-cudnn/` (CUDA + cuDNN),
  `build-ocl/` (OpenCL). The harnesses `nert_gate_{cpu,cuda,opencl}` statically
  link `libnert.a` and **must be relinked after any rebuild**; never run a GPU
  leg against a stale binary.

## Standing rules and lessons (apply from now on)

1. **torch's `dynamo` default.** `torch.onnx.export` in torch 2.12 defaults to
   `dynamo=True`, which produced a **non-self-contained** ViTPose artifact. Add
   `dynamo=False` at the call site (the ViTPose run patched a `/tmp` copy only)
   and assert self-containment in `verify_export.py`.
2. **Isolate the export environment.** Keep `HF_HOME` / `TORCH_HOME` inside
   the packet directory, and use a **venv** when the system `transformers` is
   too old (4.35.0 has no ViTPose; 4.57.1 was installed in an export-local
   venv and the system site-packages were left untouched).
3. **Acceptance asserts a captured exit status**, never a self-attested
   `EXIT=0` string grepped from worker prose.
4. **A BLOCKED probe verdict is a legitimate completed outcome**, not a failed
   packet; it files a separate kernel packet only for `dispatchable_missing`.
5. **The loader folds `Constant` at read time** (`src/model_loader.cpp:1804-1812`),
   so `Constant` is `loader_folded` and receives **no kernel row**; the
   `FoldConstants` API is unused. Attribute the fold to the loader's read loop,
   not to `FoldConstants`.
6. **A correction packet must retain a pre-edit copy of any file it edits**, so
   the reviewer can audit the diff.

## Boundary-5 order (maintainer-set)

- One packet per model family; at most two packets in flight with disjoint
  writable paths; GPU legs serialized. **Each model gets its own boundary**:
  gate → reviewer → integration → commit → **fresh orchestrator reset**; do
  not carry model N+1 into model N's context.

1. **Depth-Anything-V2-Small** — full CPU/CUDA/OpenCL **GATE** (the export and
   its provenance are already done and reviewed — see above), then reviewer,
   integration, commit, reset.
2. **RAFT-small** — full CPU/CUDA/OpenCL **GATE** (export and provenance
   already done and reviewed), then reviewer, integration, commit, reset.
   Note for the probe: the source exporter hardcodes opset 16 **and**
   `dynamic_axes`, so the shape pinning is already in the artifact.
3. **VideoMAE / ViViT / TimeSformer** — export at opset 18, probe, then gate
   each; one shared static 5D fixture frozen in the provision provenance.
4. **RF-DETR detection / segmentation / keypoint at other sizes** (entries
   22–24) — the maintainer has ruled that the track **may be attempted**: fix
   the `rfdetr` / transformers dependency in an **isolated venv** (the system
   `transformers` 4.35.0 cannot import `rfdetr`), then export and gate. The
   weights download from `storage.googleapis.com`, so the export is **not
   hermetic** — record that in the provenance.
5. **EdgeCrafter detection / segmentation / pose** (entries 19–21) — **last**:
   upstream clone, a new environment, three checkpoint downloads; the
   `orig_target_sizes [1,2]` int64 payload must be constructed and bound.

If all nine close, the scoreboard becomes **PASS 29 / Open 1 / Missing export
0 / Out of scope 2** (29 + 1 = 30 in scope); Grounding DINO stays open, so
0.2.0 still waits on the maintainer decision below.

## Grounding DINO OpenCL remedy plan (assessment only, NOT implemented)

**Maintainer decision required — put explicitly, not made for the maintainer:**
- **Option 1 — accept a documented backend GAP-with-deviation.** Grounding
  DINO's OpenCL cell stays `GAP` (measured: the handle path demands 11,028.5
  MiB on a 5,803.3 MiB device against a 1,687.9 MiB live set). A documented
  GAP is not a PASS unless the maintainer explicitly approves it for the
  release.
- **Option 2 — require the bounded remedy.** Implement per-tensor
  plain-buffer liveness: **no public-ABI change**; touches `src/executor.cpp`
  and `test/executor_test.cpp`. A bounded packet, but it **requires a
  driver-level trace** of the allocation primitive before it can be accepted
  (the standing rule: a modelled footprint does not qualify).
- The kernel-offset arena (option B in the remedy plan) is **not on the
  table**: it needs a **public-ABI change** (`TensorView.data` holds an opaque
  `cl_mem` with no offset channel) and the maintainer has **not authorised**
  it, so it is not proposed.
- **Do not implement any OpenCL fix in this phase, whichever option is
  chosen.** The gate plan for the remedy: driver-level trace first; then the
  bounded executor-only implementation plus executor tests; then re-run the
  Grounding DINO OpenCL leg (exit status, executed index, memory peak) and
  re-run the affected existing gates for byte-identical outputs. **The public
  ABI and the kernel ABI may not change without maintainer approval.**

## Housekeeping outstanding

- Delete the stray scratch files left by the export experiments:
  `/tmp/opencode/relabel_back.onnx`, `/tmp/opencode/relabel_naive.onnx`,
  `/tmp/opencode/resave_retained.onnx`, `/tmp/opencode/before_snapshot.txt`,
  `/tmp/opencode/after_snapshot.txt`.
- Keep the standing rule above: a correction packet retains a pre-edit copy of
  every file it edits so the reviewer can audit the diff.

## Session reset

The `orchestrator` agent cannot run as a subagent, so a boundary reset uses the
CLI:

```
opencode api session.create -d '{"title":"NERT catalog program — boundary 5"}'
opencode run --agent orchestrator --session <id> --auto "<handoff prompt>" &
```

The new session reads only a handoff, the ledger, the inventory and the
coverage matrix. Record the new session id in the ledger's "Sessions and
resets" table.
