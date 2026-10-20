# Handoff — NERT catalog coverage, boundary 3 of N

Date: 2026-10-20. Written by the closing orchestrator session
`ses_ea9a8c0e2ffesxGThz1IUdG8Qf`, which ran tranche 1 of the export-first
provisioning batch. Read this file, the ledger, the inventory and the coverage
matrix; nothing else is needed to continue.

**Placeholders** (the program's convention; no local home paths are recorded
here). `$NERT_REPO` = the NERT repository root; `$LOCAL_ARTIFACTS` = the local
artifacts root that holds `tritonic/`; `$HERMES_CACHE` = the cache root that
holds `cache/scratch/nert-specs-slim-2026-10-19/` with the pre-slim coverage,
inventory and ledger; `${HOME}` = the user home.

## Where the program stands

- Repo: `$NERT_REPO`, branch `main`. The working tree is **clean** at this
  handoff and in sync with `origin/main`. This boundary's commits:
  `545855a` (the tranche-1 gates and the published record), `7227c64` (the
  ledger's commit-table row), and the commit carrying this handoff.
- **Scoreboard: PASS 19 / Open 1 / Missing export 10 / Out of scope 2**
  (19 + 1 + 10 = 30 in-scope entries of 32). The one open entry is Grounding
  DINO, whose OpenCL cell is a **backend gap**.
- **0.2.0 is not release-ready**: the gate needs zero open entries *and* zero
  missing exports, and ten entries still have no ONNX export.
- Kernel tables **unchanged**: CPU 72, CUDA 72, OpenCL 68, HIP 9 (unsupported
  for 0.2.0). **No new op row was added this boundary, and the engine was not
  touched** — no source file changed.
- The coverage matrix, the census, the README and the CHANGELOG all carry the
  19/1/10/2 scoreboard.

## What closed this session

1. **YOLOv8n detector (entry 17)**, **YOLO11m-seg (entry 18)**, **OWL-ViT
   (entry 25)** and **ResNet-50 (entry 30)** are gated and recorded. Every
   device leg exited 0 with equal `nodes_dispatched` and **0 elements over the
   scaled tolerance** on every output. Dispatched counts: 260, 467, 1,071,
   122.
2. **OWL-ViT needed an architect ruling, and the ruling changed a rule.** The
   first export was valid but **unloadable**: NERT's loader rejected a `Cast`
   to 11 (DOUBLE — `include/nert/graph.hpp:42` has no `Float64`) and a
   `Reciprocal` outside the loader's accepted-op map. Both are **loader**
   rejections, before any kernel lookup, so **kernel rows could not have fixed
   it**. The architect ruled **re-export, do not touch the engine**: both
   blockers are compile-time artifacts of the export (an exact identity double
   round trip; a division by a compile-time constant whose whole chain is
   foldable), not model semantics. Two sites were patched **in the exporter
   only**; the patched artifact is proven **bit-identical** to the rejected one
   on both outputs across five input sets, with the structural diff confined to
   the two patched constructs (1,439 → 1,418 nodes, 21 removed, 0 added). The
   rejected artifact is preserved beside it and is not loadable by NERT. The
   gate's precondition — the patched artifact's ORT reference byte-identical to
   the rejected artifact's — was re-confirmed at gate time.
3. **ResNet-50 was converted**, not pinned: the Neuriplo torchvision exporter
   has **no opset argument**, so it emitted opset 20 and
   `onnx.version_converter` produced the opset-18 artifact. The provenance
   carries the verbatim caveat *"conversion does not prove semantic
   equivalence — the full-model gate is the equivalence evidence."* Two
   provenance facts the row carries: the exporter's `export_to_onnx`
   **unconditionally replaces `model.fc` with a fresh unseeded
   `torch.nn.Linear`**, so the head carries no ImageNet training (the logits
   are **not** ImageNet-classifiable, and the artifact is not
   byte-reproducible); and its pooling node is `ReduceMean`, which **differs**
   from the ResNet-18 `/tmp` export's `GlobalAveragePool`.
4. **Three factual defects in the evidence files were found by reviewers and
   corrected**: a stale `provenance.md` and `op_census.json` that described the
   *rejected* OWL-ViT artifact, an inverted ResNet-18 pooling claim, and a
   false "archive freshness" clause in the YOLOv8n evidence.

## Rules this session added (apply from now on)

The architect amended the kernel-row rule and added a fifth standing rule:

1. **A missing-row diff taken from a graph the loader rejected is not a
   missing-row diff.** The graph never reached dispatch, so no op census can be
   attributed to a device. `missing_rows.txt` must label each op
   `loader_rejected` / `loader_folded` / `dispatchable_missing`, and **only
   `dispatchable_missing` can justify a kernel packet.**
2. **The tranche budget:** the one conditional kernel packet covers a *loaded*
   graph only. A group whose first provision produced an unloadable artifact
   may be re-provisioned without consuming it (two re-provisions per group).
3. **Standing rule 5 — a de-blocking re-export is a transcription, not a
   reinterpretation.** It must be proven to compute the same function as the
   rejected artifact: bit-identical outputs on a fixture set that exercises
   every conditional branch the patch touches, an unchanged model revision and
   environment, and a structural diff confined to the blocking construct. **The
   ORT reference is computed from the same artifact, so the GATE cannot detect
   a weakened re-export; only the comparison against the rejected artifact
   can.**
4. **The PROBE method:** where the zero fill makes a declared output constant
   by construction, the probe records that fact and does not treat the
   degenerate output as evidence about the model.

The four standing rules from boundary 1 still apply: a driver-level trace for
allocation-strategy changes; a gate's acceptance must assert each leg's exit
status; a kernel difference is a defect only when a higher-precision
recomputation puts NERT farther from the exact result than the reference; and
format checks scoped to the packet's own paths with `git -C <repo>`.

## Open decisions for the maintainer

**These five are the maintainer's to make and none is settled.** They are
recorded in full, with the working assumption each batch adopted and what a
contrary ruling changes, in the ledger's "Five maintainer decisions — still
open, and still the maintainer's" section.

1. **Does the release gate admit a documented GAP-with-deviation?** Grounding
   DINO's OpenCL cell is a backend gap that no export work unblocks. Either
   the gate admits it, or 0.2.0 waits on an OpenCL allocation-strategy change
   (per-tensor plain-buffer liveness, or a kernel-offset arena) — the latter is
   post-0.2.0 work, needing its own ruling and a driver-level trace.
2. **The three blocked RF-DETR groups** (detection / segmentation / keypoint at
   other sizes): record them as `MISSING-EXPORT` with the reason (`import
   rfdetr` fails under the installed `transformers 4.35`; the weights
   auto-download, so it is not hermetic) rather than `OUT-OF-SCOPE`. The
   alternative is a transformers ≥5.x upgrade plus a network allowance, which
   carries regression risk for the 19 PASS entries.
3. **The CUDA `logits` envelope bound**: the canonicalized mean is 7.639468e-03
   against a 7.5e-03 calibration — 1.86 % over, surfaced and not smoothed. The
   bound is per-device and needs recalibration.
4. **One tranche or one group at a time**, and whether one gated representative
   per entry closes that entry. The batch ran one tranche under a two-packet
   cap and **adopted representative-per-entry as a working assumption**; that
   assumption is **recorded in the matrix and the census**, so the scoreboard's
   dependence on it is visible rather than silent.
5. **The conversion evidence bar** for converted exports (RAFT-small, and
   ResNet-50 already): the full-model gate against ORT is the equivalence
   evidence, with the caveat recorded verbatim. If a stronger bar is wanted (an
   ORT-vs-ORT pre/post-conversion diff), it must be added to the GATE template
   — the tranche-2 RAFT-small packet records **both**, so either ruling is
   satisfiable without re-gating.

## Reusable assets (do not rebuild)

- Gate harness: `/tmp/opencode/nert_gate.cpp` plus `nert_gate_cpu`,
  `nert_gate_opencl`, `nert_gate_cuda`; build scripts
  `/tmp/opencode/build_gate_{cpu,opencl,cuda}.sh`; comparison tool
  `/tmp/opencode/gate_compare.py` (report-only — its exit code is always 0, and
  for an int64 tensor it prints `over_tol`, never a `mismatched=` token).
  **Every harness binary statically links `libnert.a`: relink it after any
  rebuild**, and never run a CUDA leg against a stale binary. CUDA runs need
  `LD_LIBRARY_PATH=${HOME}/.local/lib/python3.12/site-packages/nvidia/cu13/lib:/tmp/opencode/cudnn/lib64`
  and `CUDNN_ROOT=/tmp/opencode/cudnn`.
- ORT references and the deterministic `bus.jpg` input pattern:
  `/tmp/opencode/catalog_refs/<model>/`; the fixture recipe is
  `/tmp/opencode/catalog_refs/provision_refs.py`.
- Gate evidence: `/tmp/opencode/gates/*_evidence.md`; the standing format
  exemplar is `yolo11s_evidence.md`. Analysis helpers:
  `/tmp/opencode/gates/{seg_analysis,rtv4_analysis,sn1_analysis,det_analysis}.py`.
- Export-first artifacts: `/tmp/opencode/exports/<group>/` (provision → probe
  → gate per group) and `/tmp/opencode/catalog_refs/<model>/`. The four
  tranche-1 exports are under `/tmp/opencode/exports/{yolov8n,yolo11m-seg,owlvit,resnet50}/`.
- The OWL-ViT re-export evidence: `patch_proof.md` (E1–E7), `export_facts.json`
  (the `patches` and `rejected_artifact` blocks), `repatch_accept.sh` (the
  operative 12-condition acceptance) and `d1_probe.py` in
  `/tmp/opencode/exports/owlvit/`.

## Process rules

Planner first (one packet per model or family, with writable paths, one
acceptance command, a stop condition and a reviewer). One fresh **read-only
reviewer** after every implementer and integration packet; accept only with
reviewer PASS **and** green acceptance. At most **two** packets at once, with
disjoint writable paths, and the GPU is a single-owner resource — the OpenCL
and CUDA legs share one RTX 3060. The acceptance command runs **once**, as the
final action; a failed attempt is reported, not looped; a packet failing twice
on the same item escalates to the architect. Shared files
(`src/opencl/table.cpp`, `CMakeLists.txt`, `test/CMakeLists.txt`, the coverage
matrix, the census, `CHANGELOG.md`, `README.md`) have one serialized owner,
and **the ledger is the orchestrator's** — a worker never writes it. Do not
modify `bench/`, `docs/memory-footprint-profiling.md`,
`specs/2026-10-18-memory-footprint/`, the committed readability docs or
historical specs; do not rewrite the existing spec triple; do not touch the
Neuriplo repository (running an exporter that writes into `/tmp` is fine); no
paper HIP mirror; **no new CUDA kernel except where a catalog model genuinely
needs a missing row**, and a missing row is a **separate kernel packet** with
its own test, review and re-gate, never part of an export packet.

**Two hygiene rules the reviewers kept re-finding, worth writing into the next
packet's text:** an acceptance must assert a leg's exit status **captured from
the run**, never a self-attested `EXIT=0` string grepped from worker prose; and
a delimiter count must say whether it counts **escaped** occurrences (use
`re.findall(r'(?<!\\)\|', row)`).

## Next packets — tranche 2 (the ten remaining missing-export groups)

**Seven are executable on this host**: EdgeCrafter detection / segmentation /
pose (entries 19–21), ViTPose-base-simple (26), Depth-Anything-V2 (27),
VideoMAE / ViViT / TimeSformer (29), RAFT-small (31). **Three are not** — the
RF-DETR groups (22–24) — and **no RF-DETR provision, export or install
attempt should run without an explicit maintainer ruling**; the RF track is
fully decoupled, so decision 2 above can never block the other seven groups.

The full tranche-2 decomposition — the per-group packet text, the exporter
file:line references, the wave schedule and the method changes — is in the
ledger's "Tranche 2 — planned at this boundary, not executed" section.
**Read it before dispatching.** The recommended order is: an **RF-DOC** packet
that *proves* the non-executability from verbatim transcripts (so the
maintainer can rule without blocking the rest) alongside **RAFT-PROV**; then
ViTPose-PROV + Depth-Anything-PROV; then the three video exports alongside
**EdgeCrafter-PROV** (the long pole: upstream clone, a new environment, three
checkpoint downloads); then the gates **serialized on the GPU**, shortest CPU
leg first (RAFT-small, ViTPose, Depth), then the three video models, then the
three EdgeCrafter models last, so any re-provision cycle lands at the end of
the queue; then **INT-2** alone.

**Tranche 2 must carry these method changes**: a **static 5D fixture** for the
video group (one file, frozen in the provision provenance, reused
byte-identically, zero `dim_param` on any gated artifact); an **image-pair
fixture** for RAFT-small (the exporter hardcodes **both** opset 16 **and**
`dynamic_axes**, so shapes must be pinned as well as the opset);
**multi-input binding including int64 payloads** (EdgeCrafter's
`orig_target_sizes [1,2]`); an explicit **external-data / self-containment
check** in every converted or externally-stored export's acceptance; the
**labelled missing-row diff** and standing rules 4–5 on every probe; and the
acceptance hygiene rules above. If all seven close, the scoreboard becomes
**PASS 26 / Open 1 / Missing 3 / Out of scope 2** (26+1+3 = 30).

## Session reset

The `orchestrator` agent cannot run as a subagent, so a boundary reset uses
the CLI. Both halves of that path are verified:

```
opencode api session.create -d '{"title":"NERT catalog program — boundary 4"}'
opencode run --agent orchestrator --session <id> --auto "<handoff prompt>" &
```

The new session reads only this handoff, the ledger, the inventory and the
coverage matrix. Record the new session id in the ledger.
