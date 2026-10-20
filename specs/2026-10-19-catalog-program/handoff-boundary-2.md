# Handoff — NERT catalog coverage, boundary 2 of N

Date: 2026-10-20. Written by the closing orchestrator session
`ses_eab2639b3ffeP9kkTjzb3bVpvB`, which ran the Grounding DINO family from
boundary 1. Read this file, the ledger, the inventory and the coverage matrix;
nothing else is needed to continue.

**Placeholders** (the program's convention; no local home paths are recorded
here). `$NERT_REPO` = the NERT repository root; `$LOCAL_ARTIFACTS` = the local
artifacts root that holds `tritonic/`; `$HERMES_CACHE` = the cache root that
holds `cache/scratch/nert-specs-slim-2026-10-19/` with the pre-slim coverage,
inventory and ledger; `${HOME}` = the user home.

## Where the program stands

- Repo: `$NERT_REPO`, branch `main`. The Grounding DINO, RT-DETRv4-s and
  RF-DETR seg-nano-576 families are gated, reviewed and recorded. This boundary's
  commits: `a4df628` (CUDA asymmetric-pad `Conv` support and its device
  tests), `db2f937` (the recorded closing gates — the matrix, changelog, README,
  census and ledger), and the commit carrying this handoff. The working tree is
  **clean** at this handoff.
- **Scoreboard: PASS 15 / Open 1 / Missing export 14 / Out of scope 2**
  (15 + 1 + 14 = 30 in-scope entries of 32). The one open entry is Grounding
  DINO, whose OpenCL cell is a **backend gap**.
- **0.2.0 is not release-ready**: the gate needs zero open entries *and* zero
  missing exports, and 14 entries still have no ONNX export. The next session's
  work is that batch.
- Kernel tables unchanged this session: CPU 72, CUDA 72, OpenCL 68, HIP 9
  (unsupported for 0.2.0). No new op row was added.
- **The ledger has grown well past the maintainer's slimmed form** (it now
  carries this session's packet record, reviewer verdicts and architect
  rulings). It is the live record and every verdict the maintainer asked to be
  traceable is in it, but compressing it back to a summary is a choice for the
  next boundary, not something to do silently.

## What closed this session

1. **Grounding DINO** — CPU and CUDA pass a **task-equivalence** gate: pred_boxes
   0/3600 over `1e-3(1+|ref|)`, per-box IoU min 0.99973 with 900/900, top-1
   900/900 after canonicalizing the TopK near-tie permutation, decoded detection
   set identical at the 0.25/0.25 Hugging Face proxy, one kept query with a
   1.39e-02 margin against a 5.36e-03 max score delta. The **OpenCL leg is a
   backend gap**: the handle path demands 11,028.5 MiB on a 5,803.3 MiB device
   while the model's live set is 1,687.9 MiB — not an environment limit, not a
   numeric failure.
2. **RT-DETRv4-s** — passes on CPU, OpenCL **and** CUDA. The CUDA
   asymmetric-pad `Conv` row was the last gap; the remedy is an explicit zero
   pre-pad in both CUDA `Conv` rows, the symmetric path is unchanged, and the
   guard's two arms and both spatial axes have device tests.
3. **RF-DETR seg-nano-576** — PASS with a documented **rank-99 TopK near-tie
   deviation** (architect ruling `SN-DEV`): ORT selects 182 and the GPUs 1503 on
   a 1.43e-06 tie against 4.82e-05 of fp32 reduction-order noise. Pinned to
   ORT's selection (an ORT-neutral, single-input counterfactual) all three
   devices give dets 0/400, labels 0/9,100, top-1 100/100, mask IoU min
   0.999800 at `raw>=0` (CUDA 0.984375 at `raw>=0.5`). **No engine change was
   made or is authorised.**
4. **A rejected experiment.** An OpenCL "arena" change using `clCreateSubBuffer`
   was implemented, gated and then **reverted**: on this driver a sub-buffer
   materialises as its own device allocation on first write, so per-tensor views
   commit 70x the arena, two already-gated models regressed, and six legs
   crashed at teardown. The revert restored the tree and **eight confirmation
   legs re-ran byte-identical**. Do not retry sub-buffers on this driver.

## Rules this session added (apply from now on)

- **Allocation-strategy changes need a driver-level trace**, not a modelled
  footprint: measure create / first write / release of the exact primitive, at
  the planned count and sizes, and assert the measured peak against the stated
  bound. A model is a hypothesis; a trace is evidence — and a trace that
  contradicts the model rejects the change **before** any row is re-gated.
- **A gate's acceptance must assert each leg's exit status**, not only output
  identity: a byte-identical output written before a crash is not a pass.
- **A kernel difference is a defect** only when a higher-precision
  recomputation of the same formula from the same inputs puts NERT farther from
  the exact result than the reference. If the reference's own rounding is the
  larger error, that is the accepted numerics regime: gate it with the model
  tolerance or a documented deviation, never by degrading NERT to imitate the
  reference.
- **Format checks are repo-wide**, so a packet's acceptance can fail on
  another packet's files. Scope such checks to the paths the packet owns and
  use `git -C <repo>` so a check cannot false-pass outside the repository.

## Open decisions for the maintainer

1. **Does the release gate admit a documented GAP-with-deviation?** Grounding
   DINO's OpenCL cell is a backend gap that no export work unblocks. Either the
   gate admits it, or 0.2.0 waits on an OpenCL allocation-strategy change
   (per-tensor plain-buffer liveness, or a kernel-offset arena) — the latter is
   post-0.2.0 work, needing its own ruling and a driver trace.
2. **The three blocked RF-DETR groups** (detection / segmentation / keypoint at
   other sizes): record them as `MISSING-EXPORT` with the reason (`import
   rfdetr` fails under the installed `transformers 4.35`; the weights
   auto-download, so it is not hermetic) rather than `OUT-OF-SCOPE`. The
   alternative is a transformers ≥5.x upgrade plus a network allowance, which
   carries regression risk for the 15 PASS entries.
3. **The CUDA `logits` envelope bound**: the canonicalized mean is 7.639468e-03
   against a 7.5e-03 calibration — 1.86 % over, surfaced and not smoothed. The
   bound is per-device and needs recalibration.
4. **One tranche or one group at a time** for the export batch, and whether one
   gated representative per entry closes that entry (recommended: yes).
5. **The conversion evidence bar** for converted exports (RAFT-small, possibly
   ResNet-50 / EdgeCrafter): the full-model gate against ORT is the equivalence
   evidence, or a stronger ORT-vs-ORT pre/post-conversion diff is wanted —
   decide **before** tranche 1 runs, not after.

## Reusable assets (do not rebuild)

- Gate harness: `/tmp/opencode/nert_gate.cpp` plus `nert_gate_cpu`,
  `nert_gate_opencl`, `nert_gate_cuda`; build scripts
  `/tmp/opencode/build_gate_{cpu,opencl,cuda}.sh`; comparison tool
  `/tmp/opencode/gate_compare.py` (report-only — its exit code is always 0, and
  for an int64 tensor it prints `over_tol`, never a `mismatched=` token). CUDA
  runs need `LD_LIBRARY_PATH=${HOME}/.local/lib/python3.12/site-packages/nvidia/cu13/lib:/tmp/opencode/cudnn/lib64`
  and `CUDNN_ROOT=/tmp/opencode/cudnn`. **Every harness binary statically links
  `libnert.a`: relink it after any rebuild.**
- ORT references and the deterministic `bus.jpg` input pattern:
  `/tmp/opencode/catalog_refs/<model>/`.
- Gate evidence: `/tmp/opencode/gates/*_evidence.md`; Grounding DINO artifacts
  under `/tmp/opencode/gdino/` — `gdocl-evidence.md` (OpenCL leg),
  `gdtg_evidence.md` (task gate), `gd2cv-verify-evidence.md` (the reverted
  experiment), `gdfin2-cpu-evidence.md` (final CPU leg).
- Export-first artifacts: `/tmp/opencode/exports/<group>/` (provision → probe →
  gate per group) and `/tmp/opencode/catalog_refs/<model>/`.

## Process rules

Planner first (one packet per model or family, with writable paths, one
acceptance command, a stop condition and a reviewer). One fresh **read-only
reviewer** after every implementer and integration packet; accept only with
reviewer PASS **and** green acceptance. At most **two** packets at once, with
disjoint writable paths, and the GPU is a single-owner resource — the OpenCL and
CUDA legs share one RTX 3060. The acceptance command runs **once**, as the
final action; a failed attempt is reported, not looped; a packet failing twice
on the same item escalates to the architect. Shared files
(`src/opencl/table.cpp`, `CMakeLists.txt`, `test/CMakeLists.txt`, the coverage
matrix, `CHANGELOG.md`) have one serialized owner. Do not modify `bench/`,
`docs/memory-footprint-profiling.md`, `specs/2026-10-18-memory-footprint/`, the
committed readability docs or historical specs; do not rewrite the existing spec
triple; do not touch the Neuriplo repository (running an exporter that writes
into `/tmp` is fine); no paper HIP mirror; no new CUDA kernel except where a
catalog model genuinely needs a missing row.

## Next packets — the export-first batch (14 missing entries)

**Export first, then probe, then gate, then review.** NERT loads exactly opset 18;
pin the export at 18, or convert and record that **conversion does not prove
semantic equivalence** (the full-model gate is the equivalence evidence). A
probe loads the export through NERT's own loader and shape inference, records
the post-constant-folding node count and the op-type census, and diffs it
against the kernel tables to get the exact missing rows. The OpenCL table lacks
`Abs`, `Floor`, `GreaterOrEqual` and `Neg` — none used by a currently loadable
entry, so a new export may need them; a missing row is a **separate kernel
packet** with its own test, review and re-gate, never part of an export packet.

**Tranche 1 (these four are independent; only one gate may hold the GPU):**

| group | exporter | checkpoint | notes |
|---|---|---|---|
| YOLO11m-seg | Ultralytics CLI, per `NT/export/segmentation/InstanceSegmentation.md:1-12` | **local**: `$LOCAL_ARTIFACTS/tritonic/yolo11m-seg.pt` | offline; the local `.onnx` is opset 20 and unusable — export at 18 |
| YOLOv8n detector | `NT/export/detection/yolo/export.py:281`, `--opset 18` | downloadable | representative of the YOLO detector variants |
| OWL-ViT | **no exporter in the repo** — a new worker-written script under `/tmp` | downloadable (HF `google/owlvit-base-patch32`) | transformers 4.35 ships `OwlViT`; no dependency upgrade needed |
| ResNet-50 | `NT/export/classification/torchvision/export_torchvision_classifier.py:38-46` — **no opset argument**, so verify the emitted opset and convert if needed | downloadable | representative of the classification group |

Remaining groups: the EdgeCrafter detection / segmentation / pose triples (no
exporter in the repo; upstream clone + checkpoint download), ViTPose,
Depth-Anything-V2, VideoMAE / ViViT / TimeSformer, the YOLO v5–v12 variants,
and RAFT-small (`NT/export/optical_flow/raft/raft_exporter.py:174` hardcodes
opset 16 — export then convert to 18). The three RF-DETR groups are **not
executable here** (see the maintainer decisions). The full feasibility table,
the per-group packet text and the probe/gate acceptance commands are in the
ledger's "Prepared decomposition — the export-first provisioning batch" section.

## Session reset

The `orchestrator` agent cannot run as a subagent, so a boundary reset uses the
CLI. Both halves of that path are verified:

```
opencode api session.create -d '{"title":"NERT catalog program — boundary 3"}'
opencode run --agent orchestrator --session <id> --auto "<handoff prompt>" &
```

The new session reads only this handoff, the ledger, the inventory and the
coverage matrix. Record the new session id in the ledger. The session created
for this boundary is `ses_ea9a8c0e2ffesxGThz1IUdG8Qf`.
