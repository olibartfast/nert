# Catalog program — orchestration ledger

Opened 2026-10-19. Release goal: the complete Neuriplo tasks catalog — 30
in-scope ONNX entries out of 32. Image understanding (GGUF/llama.cpp) and
Gaussian Splatting (LGM/GRM) are maintainer-approved out of scope. The
measured inventory is [inventory.md](inventory.md); the living device matrix
is [coverage.md](../2026-10-17-task-models/coverage.md).

## Contract

- **Acceptance:** the packet's single command, run once as the final action.
  Kernel packets build and test the OpenCL tree; integration packets also
  update coverage and the changelog.
- **Roles:** planner → fresh implementer → fresh read-only reviewer →
  orchestrator. A packet is accepted only with reviewer PASS and green
  acceptance. Gate runners verify outputs; they are not reviewers.
- **Concurrency:** at most two packets, only with disjoint writable paths.
  Shared files have one serialized integration owner.
- **Context reset:** at a model boundary (kernels + gate + reviewer + commit),
  write a handoff and start a fresh orchestrator session. Compaction is not a
  reset.

## Status

| Item | State |
| --- | --- |
| CPU / CUDA kernel tables | 72 / 72 rows |
| OpenCL kernel table | 68 / 72 rows; the four absent types are unused by currently loadable catalog entries |
| HIP | 9 rows; unsupported for 0.2.0 |
| Recorded full-model PASS | 19 of 30 in-scope entries on CPU/CUDA/OpenCL (two carry the documented R-6 sub-0.001 TopK deviation, one the documented rank-99 TopK near-tie deviation) |
| Open, in scope | 1 entry (Grounding DINO, OpenCL backend gap) |
| Missing ONNX export | 10 entries |
| Out of scope | 2 entries |
| Release | **not ready**; 0.2.0 is unreleased |

The 19 recorded PASS entries are ResNet-18, YOLO26n, OWLv2, RF-DETR
seg-small, RAFT-large, YOLO11s, YOLOv8n-pose, YOLO26s-pose, RF-DETR
nano-1101, YOLO26n-depth, RF-DETR keypoint-preview, YOLO26s, YOLO26x,
RT-DETRv4-s, RF-DETR seg-nano-576, **YOLOv8n, YOLO11m-seg, OWL-ViT and
ResNet-50** (the tranche-1 export-first batch, recorded below). The one open
entry is:

- **Grounding DINO** — CPU/CUDA task-equivalence PASS (the decoded-detection set
  check closed 2026-10-20: pred_boxes 0/3600 canonicalized, per-box IoU min
  0.99973 with 900/900, top-1 900/900, decoded set_diff 0); the **OpenCL cell
  is a documented backend GAP** after the GD-2c arena-view experiment was
  reverted (the handle path demands 11,028.5 MiB on a 5,803.3 MiB device against
  a 1,687.9 MiB live set). Whether the release gate admits a documented
  GAP-with-deviation is an open maintainer decision.

Closed this session: RF-DETR seg-nano-576 (PASS with the documented rank-99
TopK near-tie deviation, `SN-DEV`) and RT-DETRv4-s (PASS on all three devices
after the accepted CUDA asymmetric-pad `Conv` remedy). **The census
(`inventory.md`) carries the same 15/1/14/2 scoreboard and the same three
entry rows, corrected by the orchestrator to match these closed gates.**

**Boundary 3 closed four more entries** by the export-first method — YOLOv8n
(entry 17), YOLO11m-seg (18), OWL-ViT (25), ResNet-50 (30) — taking the
scoreboard to **19 / 1 / 10 / 2**. **The census and the coverage matrix now
carry that scoreboard.** No release was cut: the gate still requires zero
open entries and zero missing exports among the 30 in-scope entries.

**Note for the next session:** the `## Definitions` block in `coverage.md`
defines `GAP` as "at least one op type used by the model has no row on that
device", while Grounding DINO's OpenCL cell is a `GAP` for a **capacity** reason
(all 26 op types it uses are present). The cell explains itself, but the
definition and the cell disagree; the packet was forbidden to change
`Definitions`, so this is a maintainer decision.

## Run history

| Phase | Rows | Outcome |
| --- | ---: | --- |
| P0 OpenCL rows and primary gates | 1–23 | Table 37 → 55 → 68; YOLO26n, RF-DETR seg-small, RAFT-large pass OpenCL; int64 `Clip` added |
| Review and FIX-1 | 24–30 | Reviewer rejected the 13-row slice; ScatterND/Or/ReduceMean/Identity/Squeeze defects fixed; `346cb29` |
| Grounding DINO family | 31–45 | Vision-encoder bisect; OpenCL allocation diagnosis; GD-2c arena-view experiment; reviewer found the seg-nano/OWLv2 regressions |
| GD-2c revert and RTV4 remedy | 46–53 | Arena-view path reverted; eight drifted legs restored; OWLv2 PASS restored; RTV4 CUDA remedy accepted; tree back to HEAD + ledger |

The row-by-row packet text, architect rulings and reviewer findings are
historical detail. The acceptance evidence and per-model numbers are in the
linked coverage matrix and the dated validation packets; the full pre-slim
ledger is retained at
`$HERMES_CACHE/cache/scratch/nert-specs-slim-2026-10-19/ledger-full.md`
on the maintainer host.

## Commits

| Commit | Content |
| --- | --- |
| `6d99da9` | 18 OpenCL rows (37 → 55); YOLO26n / RF-DETR seg-small / RAFT-large OpenCL gates; int64 Clip CPU/CUDA |
| `b905792` | 13 more OpenCL rows (55 → 68); coverage, ledger and changelog |
| `346cb29` | FIX-1: ScatterND ordering, Or arity, ReduceMean noop, Identity/Squeeze rank cap; catalog gate evidence; HIP unsupported |
| `14cbde8` | CUDA asymmetric-pad `Conv` support (explicit zero pre-pad in both CUDA Conv rows) + its device tests; RT-DETRv4-s's full-model CUDA gate |
| `8a00aab` | The closing gates: Grounding DINO (CPU/CUDA task-equivalence, OpenCL backend gap), RF-DETR seg-nano-576 (SN-DEV deviation), RT-DETRv4-s CUDA; matrix, changelog, README, census and ledger |

## Sessions and resets

| Scope | Orchestrator session |
| --- | --- |
| Rows 1–26 | `ses_eaca1c986ffeFJdfdPOFfK3mDh` |
| Rows 31+ (boundary 1) | `ses_eab2639b3ffeP9kkTjzb3bVpvB` |
| Export-first batch (boundary 3) | `ses_ea9a8c0e2ffesxGThz1IUdG8Qf` |

Boundary handoff: [handoff-boundary-1.md](handoff-boundary-1.md). The fresh
session reads only the handoff, this ledger, the inventory and the coverage
matrix. The orchestrator agent cannot run as a subagent; resets use the CLI
session create + run path.

## Open work

1. Close the Grounding DINO boundary: run the decoded-detection set check,
   record CPU/CUDA as PASS (GD deviation), record OpenCL as a backend GAP, and
   do not claim PASS from op coverage.
2. ~~Run the RT-DETRv4-s full-model CUDA gate after the asymmetric-pad Conv
   remedy.~~ **Done** — 996 nodes, exit 0, boxes 0/1200, scores 0/300, labels
   300/300 exact, 11 detections over 0.25 with class ids
   `[5, 0, 0, 0, 0, 26, 9, 11, 26, 26, 26]`.
3. Resolve or reclassify RF-DETR seg-nano-576's TopK tie with the export/
   topology owner.
4. ~~Provision the 14 missing ONNX exports export-first, then probe, gate and
   review each model.~~ **Tranche 1 closed 4 of them** — YOLOv8n (entry 17),
   YOLO11m-seg (18), OWL-ViT (25), ResNet-50 (30). **Ten remain:**
   EdgeCrafter detection / segmentation / pose, RF-DETR detection /
   segmentation / keypoint at other sizes, ViTPose-base-simple,
   Depth-Anything-V2, VideoMAE/ViViT/TimeSformer, RAFT-small. No release
   until the in-scope scoreboard is clean.

### State of the in-flight packets (2026-10-19, boundary 1 session)

The spec documents were slimmed by the maintainer mid-session; the row-by-row
packet text, architect rulings and reviewer findings for rows 31-53 are in the
pre-slim archive named above, and the acceptance evidence is in the dated
validation packets and the `/tmp` gate artifacts. What follows is the state a
fresh session needs, not a transcript.

- **Grounding DINO.** The CPU and CUDA legs are re-measured on the final tree:
  CPU pending, CUDA **PASS** (19,224 nodes, wall 32:01, logits max 1.607923 /
  mean 1.373600e-02, 48,673 of 230,400 over `1e-3(1+|ref|)`; pred_boxes 80 of
  3,600; a YOLO11s CUDA spot-check is byte-identical to its pre-fix artifact).
  The task-equivalence verdict, the decoded-detection set check and the
  per-device envelope are the **task-gate packet**'s job; it must derive the
  TopK permutation from the fresh outputs rather than reuse a hard-coded list.
  OpenCL is a **backend GAP**, not an environment limit: the handle path
  allocates one buffer per distinct offset eagerly (1,303 buffers / 10,101 MiB
  against a 1,687.9 MiB live set on a 5,668 MiB device).
- **The GD-2c arena-view experiment was reverted and the revert verified.**
  A sub-buffer is a *new* device allocation on first write on this driver
  (measured: 905 MiB parent, 1,000 sub-buffers, 131 → 5,791 MiB on first
  write, 566/1,000 writes succeeding), so per-tensor views commit the sum of
  written sizes — 70x the arena for Grounding DINO. It also regressed OWLv2
  and crashed six legs at teardown. The revert restored the tree to HEAD plus
  the accepted RTV4 remedy; **eight confirmation legs re-ran exit 0 and
  byte-identical**, OWLv2's PASS and the seg-nano failure signature included,
  and the teardown SIGSEGV is gone from **those eight logs** (the GD-2c logs
  do contain it — the earlier phrasing "no `signal 11` in any log" was too
  wide and is corrected here). **Do not reintroduce `clCreateSubBuffer` views
  on this driver.**
  **The eight "OK" lines are not eight PASSes — three classes, and the coverage
  matrix must keep them apart** (from the revert's review, session
  `ses_ea15135affeUs0KPBKCCgEi5N`, verdict PASS-WITH-FINDINGS, which
  independently `hash-object`-verified all 11 files against HEAD, `cmp`-ed
  every artifact and re-derived the seg-nano metrics from the ORT references):
  - **exact byte-identical PASS** — OWLv2, RF-DETR seg-small, YOLOv8n-pose,
    YOLO26s-pose, RF-DETR nano-1101;
  - **byte-identical with the known R-6 deviation, not PASS** — YOLO26s and
    YOLO26x, both 10/1,800 over tolerance (max abs 54 / 182.17; max ratio
    18,000 / 3,500);
  - **a known numeric failure, not a crash** — seg-nano-576, producing output
    identical to its pre-fix failure artifact (dets 50/400, labels 3,628/9,100,
    masks 1,239,727/2,073,600; top-1 99/100 with only query 99 flipping
    64→84; TopK rank-99 182 vs 1503, one differing position).
  **"Byte-identical" must not be promoted to "PASS" in the matrix.**
  Three further corrections from that review: (a) the "new in GD-2c" attribution
  is **per row** — raw-proven for YOLO26s/26x, documented in the pre-fix
  evidence for seg-small and YOLOv8n-pose, and **formally open for YOLO26s-pose
  and RF-DETR nano-1101** (their pre-fix evidence records completion but no
  exit code); the *absence* of the crash is closed for all six by the exit-0
  logs. (b) The OWLv2 runner wrote ten tensors under `/tmp/opencode/owlv2-data/`,
  not only the two safeguarded ones — the runner hardcodes its output paths —
  so the footprint is wider than the scratch prefix; no tracked file was
  touched and the two baselines are intact. (c) The architect's gdb result
  (212 of 399 sub-buffers released, failing on the 213th, inside
  `clReleaseMemObject`) has **no transcript in the artifacts** — it is
  architect-supplied context, and the sub-buffer path is verifiably absent
  from HEAD, so the remediation does not depend on it. The seg-nano metrics
  artifact also contains a misleading first line (90/91) — cite 99/100.
  **Still open: Grounding DINO's OpenCL leg was not re-measured on the reverted
  tree** by the revert packet, so the OpenCL GAP must not be recorded as
  measured until a leg on the reverted tree captures the status, node index and
  memory peak.
- **RT-DETRv4-s — the CUDA gap is closed.** The asymmetric-pad `Conv` remedy
  is accepted (24/24 `build-cudnn`, `nvcc` translation-unit check clean,
  symmetric path proven byte-identical) and the **full-model CUDA gate passed**
  (996 nodes, exit 0, wall 98.6 s; boxes 0 of 1,200 at max abs 7.93457e-04,
  scores 0 of 300 at 2.45981e-05, labels 300 of 300 exact, 11 detections over
  0.25 with class ids `[5, 0, 0, 0, 0, 26, 9, 11, 26, 26, 26]`). Neither the
  `asymmetric padding` nor the `negative padding` guard fired, as expected —
  exactly one Conv in the graph needs the new path. `coverage.md`'s
  RT-DETRv4-s row (`PASS | GAP | PASS`) is now stale in its CUDA cell and the
  integration packet must move it to `PASS`. Two caveats to carry:
  `build-cudnn` has `NERT_CUDNN_FRONTEND_DIR` empty, so the live path is the
  legacy `cudnnConvolutionForward` row and `conv_frontend.cu` is **not
  exercised by this model** — it remains covered only by the `nvcc`
  translation-unit check; and `gate_compare.py` emits `over_tol` for int64 but
  **never** a `mismatched=` token (that line came from a separate, labelled
  numpy recount), so no acceptance command should require `mismatched=` to
  come from `gate_compare.py`.
- **RF-DETR seg-nano-576 — the signature is re-established on the current
  tree (SN-1).** All three legs exit 0 with 1,216 nodes: CPU (488.96 s) `dets`
  0/400, `labels` 0/9,100, `masks` **2**/2,073,600 at max ratio 1.27923 (CPU
  marginals, not "0 over tolerance"), top-1 **100/100**, `TopK idx[99]=182`
  matching ORT, swap count 0; CUDA (4.67 s) `dets` 50/400, `labels`
  3,629/9,100, `masks` 1,240,141/2,073,600, top-1 99/100 with the single
  mismatch at query 99 (reference class 64 → device class 84), `idx[99]=1503`,
  differing positions **[99]**; OpenCL (3.27 s) the same with `labels` 3,628
  and `masks` 1,239,727. The OpenCL outputs are **byte-identical** to the
  post-revert `oclv_r_*` baseline, and the CPU and CUDA outputs are
  byte-identical to their pre-fix artifacts, so the tree change did not move
  any leg. The TopK indices came from **read-only output-declaring probe
  models** whose `dets`/`labels`/`masks` are byte-identical to the gate legs',
  proving the probes did not perturb the computation; the pre-existing
  `seg_nano_topk_pinned.onnx` was not used, edited or overwritten. **The
  decisive refinement for the disposition:** the noise is **comparable on all
  three devices** (max |ORT − device| 9.39369e-05 CPU = 65.7x the margin,
  1.24216e-04 OpenCL = 86.8x, 4.81606e-05 CUDA = 33.7x, against a reference
  margin of 1.43051e-06) — the CPU survives because **its own** margin is
  1.67x the reference's (+2.38419e-06), not because it is quieter. The
  remaining question is the disposition of the near-tie.
- **The seg-nano counterfactual is measured (SN-2).** ORT-neutrality is
  **triple-zero** (`max_abs 0` on `dets`, `labels` and `masks`), so the pin is
  semantically neutral and the comparison is meaningful. The fresh derivation
  replaced exactly one node (`/transformer/TopK` → two `Constant` nodes holding
  ORT's TopK values and indices; every other node byte-identical; 1,215
  dispatched vs 1,216) and came out **bit-identical to the pre-existing
  `seg_nano_topk_pinned.onnx`** (same sha256 `db20d63b…`) — an independent
  reproduction, not a copy; the pre-existing file's sha256 and mtime are
  unchanged. The three pinned legs all exit 0 with **`dets` 0/400, `labels`
  0/9,100 and top-1 100/100 on every device**, `masks` at CPU-parity
  marginals (CPU 2, OpenCL 36, CUDA 17 of 2,073,600; max ratio ≤ 2.6) and mask
  IoU min **0.999800** at the `raw>=0` threshold on all three; at `raw>=0.5` the
  CUDA leg's min is **0.984375** (one query) and is disclosed verbatim, with
  CPU and OpenCL at 1.000000. **With the TopK pinned to ORT's, all three
  devices match the reference** — that is the measured basis of the proposed
  deviation. The `SN-DEV` ruling request is carried verbatim in
  `sn2_evidence.md`; the alternative export/topology path is documented as
  **not executable here** (`import rfdetr` fails under `transformers 4.35.0`,
  and rfdetr 1.8.0 downloads checkpoints from `storage.googleapis.com`, so it is
  not hermetic), and the Neuriplo repository must not be modified either way.
- **Scoreboard arithmetic to get right at integration** (the planner's draft
  would have closed all three): the coverage definitions count a PASS only
  when the gate is recorded "on all three listed devices", so Grounding DINO —
  `PASS (GD deviation) | PASS (GD deviation) | GAP` — **remains an open entry**
  (its OpenCL cell is a backend GAP), while seg-nano-576 and RT-DETRv4-s become
  full PASS entries. The grant case is therefore **PASS 15 / Open 1 (Grounding
  DINO) / Missing 14 / Out of scope 2**, and 15 + 1 + 14 = 30 in scope. The
  release gate is still not met — the 14 missing exports are the next
  session's work — which is the maintainer's own standing statement.
- **Standing rules added this session, each from a corrected claim:** a kernel
  difference is a defect only when a higher-precision recomputation puts NERT
  farther from the exact result than the reference; an allocation-strategy
  change needs a **driver-level trace** of the primitive it depends on, not a
  modelled footprint; and a gate's acceptance must assert each leg's **exit
  status**, not only output identity.
- **The maintainer edited the specs and committed mid-session.** `f2f12c9`
  ("docs(readme): update release and device coverage status") landed at 21:09;
  the slimming of `coverage.md`, `inventory.md` and this ledger is still
  uncommitted in the working tree, and `README.md` was edited again after it.
  The maintainer also pre-seeded `coverage.md` with the states these packets
  established: Grounding DINO `PASS (GD deviation)*` / `PASS (GD deviation)*` /
  `GAP`, RT-DETRv4-s `PASS | GAP | PASS`, seg-nano-576 `PASS | FAIL | FAIL`.
  **Working rule for the rest of this session:** the spec documents are now
  summary documents, not transcripts; the row-by-row text and the D1-D18
  correction series live only in the pre-slim archive, and new evidence goes
  into a coverage Evidence cell and the ledger's status/open-work sections. Any
  edit to them must re-read the file immediately beforehand and anchor on
  **content, not line numbers** (the older line references are stale), and must
  not revert the maintainer's slimming.
- **The CPU leg is being re-run.** The first CPU leg (session
  `ses_ea98ba31ffexhJdhExEcpCPRU`) is sound — exit 0, 19,224 nodes, wall
  1:21:48, logits max 1.606933 / mean 1.323906e-02 / 47,138 over
  `1e-3(1+|ref|)`, pred_boxes 72 of 3,600, zero non-finites, and a
  byte-identical YOLO26n CPU spot-check — but it linked `build/libnert.a` from
  20:23, **before** the arena-view revert at 20:59, so it was measured against
  a library that has since been replaced. Its acceptance also exited 1 on the
  repo-cleanliness grep because the maintainer's commit moved HEAD underneath
  it. A re-run against the final code tree is in flight (session
  `ses_eaa4499beffeYDFAJIYsa4gxBC`); it also compares its outputs
  byte-for-byte against the first attempt's, which is what reconciles the two.
  The CPU is otherwise idle, so the re-run costs no wall-clock on the GPU
  track.
- **RT-DETRv4-s's full-model CUDA gate is in flight** (session
  `ses_ea727699ffek8pcBWfa3w5FCL` is the CUDA Grounding DINO leg; the
  RT-DETRv4 gate follows). Its acceptance asserts the leg's **exit status and
  node count (996)**, not only the compared values.

- **SN-DEV ruling (2026-10-19, architect): GRANTED WITH CONDITION.** RF-DETR seg-nano-576 is recorded `PASS | PASS (SN deviation) | PASS (SN deviation)`. Basis: SN-1/SN-2 at `f2f12c9` — one rank-99 TopK near-tie (ORT 182, both GPUs 1503; ORT gap 1.43051e-06 vs max ORT−device deviation 4.81606e-05 CUDA / 1.24216e-04 OpenCL; CPU is not quieter at 9.39369e-05 and keeps 182 only because its own margin is +2.38419e-06, 1.67× the reference's). NERT's selection rule already reproduces ORT's indices on ORT scores and the GPU's on GPU scores, so **no engine change is required or authorised**; a TopK tie-break change is not permitted under this deviation. The ORT-neutral pin (single `TopK`→2×`Constant` substitution, sha256 `db20d63b…`, max_abs 0 on `dets`/`labels`/`masks`) gives dets 0/400, labels 0/9,100, top-1 100/100 on all three, masks 2/36/17 of 2,073,600, IoU min 0.999800 at `raw>=0`. Condition: the coverage row must carry the pinned basis, the unpinned query-99 flip and over-tolerance counts, and the CUDA `IoU@raw>=0.5` min **0.984375** (query 36, 63/64); a bare PASS or any implication of unpinned output equality voids the grant. Revisit on any change to the export (`300900e3…`), the reference (ORT 1.27.0 CPU), the tolerance/bar, or the TopK selection/tie-break. Scoreboard after integration: **PASS 15 / Open 1 (Grounding DINO) / Missing export 14 / Out of scope 2**, provided the already-measured RT-DETRv4-s CUDA cell and the asymmetric-pad remedy it rests on are committed in the same integration; a seg-nano-only integration is 14 / 2 / 14 / 2. This closes Open Work #3.

### The SN-DEV ruling, in the maintainer's compact terms

- **What is granted.** seg-nano-576's CUDA and OpenCL cells become
  `PASS (SN deviation)`. The CPU cell stays an unqualified `PASS`, because it
  reproduces ORT's rank-99 selection **unpinned** (`idx[99]=182`, swap count 0,
  top-1 100/100).
- **The exact row the integration must write** (the architect rejected the
  proposed wording as too thin — it omitted the unpinned counts and the CUDA
  0.984375, and did not say the PASS rests on the pin):

  ```
  | RF-DETR seg-nano-576 | detection + segmentation | PASS | PASS (SN deviation) | PASS (SN deviation) | Rank-99 TopK near-tie (SN-DEV): ORT 182 vs GPU 1503; ORT gap 1.43051e-06 vs max score-vector ORT−device deviation 4.81606e-05 CUDA / 1.24216e-04 OpenCL on `/transformer/ReduceMax_output_0` (2,304 elements). Selection rule already ORT's; no engine change. Unpinned, both GPUs: query 99 flips 64→84, top-1 99/100; dets 50/400; labels 3,629 CUDA / 3,628 OpenCL of 9,100; masks 1,240,141 CUDA / 1,239,727 OpenCL of 2,073,600 over tolerance (max ratio 47,350.3 / 47,350.4). PASS is on the ORT-neutral pin (TopK→2×Constant; ORT max_abs 0): dets 0/400, labels 0/9,100, top-1 100/100 on all three; masks 2/36/17 (CPU/OpenCL/CUDA) of 2,073,600, max ratio ≤ 2.6; IoU min 0.999800 at raw>=0; CUDA 0.984375 at raw>=0.5 (query 36, 63/64; CPU/OpenCL 1.000000). The pinned graph is a single-input counterfactual for this gate (TopK removed; 1,215 vs 1,216 nodes): not a TopK-correctness result, not a deployable substitute. Result-level equivalence, not bit-identity. |
  ```

  The Evidence cell is the fifth cell verbatim. Do **not** change the
  Definitions section and do **not** introduce a new status token such as
  "CONDITIONAL PASS" — the existing `PASS (…deviation)` convention is the
  contract. In Open Items, item 2 becomes "Closed by SN-DEV ruling
  (2026-10-19); see ledger", and the stale RT-DETRv4 item 3 is removed when
  that row is integrated.
- **The CUDA `IoU@raw>=0.5` min of 0.984375 is inside the deviation.** The
  release binarisation is `raw>=0` (the sigmoid-0.5 boundary), where all three
  devices are 0.999800; the 0.984375 is one query (query 36, inter 63 / union
  64), all 100 queries remain ≥ 0.9 at both thresholds, and the elementwise
  gate has only 17 of 2,073,600 mask elements over tolerance at max ratio
  1.25271. It is **not** a semantic mask failure, but it must be recorded
  **verbatim** — not rounded to "1.0" and not omitted. If a release bar of
  `IoU >= 0.999` at `raw>=0.5` is ever adopted, the grant is reopened.
- **What voids the grant.** A bare `PASS` on either GPU cell, omission of the
  unpinned query-99 signature, of the **unpinned masks over-tolerance counts
  (1,240,141 CUDA / 1,239,727 OpenCL of 2,073,600, max ratio 47,350.3 /
  47,350.4)**, of the unpinned dets/labels counts, or of the CUDA 0.984375, or
  any wording implying unpinned or bit-for-bit agreement. The reviewer must reject the row if so; it
  reverts to `PASS | FAIL | FAIL`.
- **No engine-side change is permitted under this deviation.** A future
  epsilon or near-tie tie-break is not allowed: the GPU values are strictly
  ordered, so it would fabricate a tie band of at least 3.58e-06, be arbitrary,
  and change the selection on other near-tie models (YOLO26s/x's R-6 deviation
  and the byte-exact `oclv_r_*` OpenCL baseline). That would need a separate
  architect ruling and a full re-gate.
- **Revisit conditions.** The grant is not a standing waiver. It is revisited
  on any change to (1) the export `rfdetr_seg_nano_576-18.onnx`
  `sha256 300900e355d43d6e410bca3e79f6eb38019e2b56cfc79b88ed62f4d0c0e0e0e8`,
  including a re-export that pins TopK or raises precision; (2) the reference
  (ORT 1.27.0 CPUExecutionProvider) or the input; (3) the tolerance or bar
  (`--tol 1e-3` scaled, the mask IoU bar, a `raw>=0.5` release bar);
  (4) the engine's TopK selection or tie-break, or a reduction-order change
  that moves the rank-99 scores; (5) the counterfactual — if the pin stops
  being ORT-neutral, or CUDA and OpenCL stop agreeing with each other.
- **Two risks named by the architect that the integration must not create.**
  *Precedent creep:* "SN deviation" must not become a generic waiver — the
  ledger line keys it to one export hash, one rank, a below-fp32 gap and a
  named counterfactual, and any other model must earn its own ruling.
  *CPU misread as the clean oracle:* the CPU has 2 of 2,073,600 mask elements
  over tolerance, so the row must not describe CPU as bit-identical or
  quieter. **Correction to the architect's own parenthetical** (from the SN-1/SN-2
  review): the CPU does **not** have the largest absolute score deviation —
  **OpenCL does** (1.24216e-04 against the CPU's 9.39369e-05, with CUDA
  smallest at 4.81606e-05). The CPU's max score-vector deviation is still
  **65.7x the reference margin**, and its own max noise is **39.4x its own
  margin**; it keeps index 182 because **its own** scores order 182 above 1503
  by +2.38419e-06. The integration must state those facts, not the mislabel.
- **The RTV4-1 test gap is closed.** The follow-up added two device tests to
  `test/cuda_conv_test.cpp` (+85 lines, **zero deletions**; 24/24 `build-cudnn`
  ctest, format clean): `NertCudaConv.AsymmetricBeginPadMixedWithBias` uses
  `pads=[1,0,0,2]` → `pad_begin=[1,0]`, `pad_end=[0,2]`, so the
  `srcOff[i]=pad_begin[i]` path is now exercised **on device** — the two
  pre-existing asymmetric cases both resolved `begin=[0,0]`, so they could not
  discriminate a remedy that ignored the begin pad; and
  `NertCudaConv.RejectsNegativePadding` uses `pads=[-1,0,0,1]` with an
  `EXPECT_THROW` pinning both the type (`nert::InferenceException`) and the
  message `negative padding is not supported on CUDA here`. Both are in the
  file's shared `NertCudaConv` suite and were proven to execute (12 tests from
  1 suite; both names in the listing and the verbose output). The
  implementation was not touched.
- **The RTV4-1 test gap is reviewed and closed.** The follow-up's review
  (session `ses_ea154c01ffeD9YH7AFubTTMnv`, verdict **PASS-WITH-FINDINGS**)
  confirms **F1 and F2 are both closed**, and verifies the discrimination
  independently rather than accepting the worker's argument: for the
  `pads=[1,0,0,2]` geometry the CPU oracle at `[oc=0, oh=0, ow=0]` is
  **1.125**, while a remedy that copies the input at offset `(0,0)` instead of
  `(1,0)` produces **10.5** at the same element — `|diff| = 9.375` against the
  `1e-3·(1+|ref|) = 0.002125` tolerance, with **18 of 24** outputs over
  tolerance and a maximum difference of 11.25. A remedy ignoring `pad_begin`
  entirely would also fail the implementation's own output-shape check. The
  geometry is genuinely discriminating, and the pre-remedy row throws before
  any cuDNN call. The negative-pad case resolves to a negative `pad_begin` with
  `prePad` true and the test's output dims matching the shape formula, so the
  **guard** fires rather than a shape error, and the test pins both the
  `nert::InferenceException` type and the exact guard phrase. The diff is a
  single hunk, **174 additions and 0 deletions**; the first 89 added lines are
  byte-identical to the accepted RTV4-1 blob (`36cc6529…`) and the source
  blobs still match the RTV4-1 post-images exactly, so the implementation is
  untouched.
  Three narrow residuals, recorded rather than waved away: the width-axis
  begin pad is still zero in all three asymmetric tests, so a regression
  confined to the inner width offset would pass; the negative-**end**-pad arm
  of the guard is untested (only the negative-begin arm is); and the test pins
  the guard phrase but not the node-name context. The evidence narrative cites
  **development** logs (`rtv4c_conv_verbose.log` 22:45:38,
  `rtv4c_full_ctest.log` 22:45:34) that predate the final binary (22:46:01);
  the authoritative ctest evidence is
  `build-cudnn/Testing/Temporary/LastTest.log` (22:46:14, 24 "Test Passed"
  entries, the `nert_cuda_conv` block running 12 tests, 12.49 s). The "234
  files" format count and the combined acceptance exit status are not
  independently reproducible from a saved log, though the changed file is
  `clang-format-18` clean and the final ctest log shows everything passing.
  The test file is **ready to commit** with the remedy.
- **The guard's remaining axes are now covered.** The residual-coverage packet
  added two more device cases to `test/cuda_conv_test.cpp` (263 additions, **zero
  deletions**; 24/24 `build-cudnn` ctest, 14 tests in the `NertCudaConv` suite,
  format clean, evidence citing the **final** `LastTest.log` at 22:55:54):
  `AsymmetricWidthBeginPadWithBias` uses `pads=[0,1,0,0]` →
  `pad_begin=[0,1]`, `pad_end=[0,0]`, so the **width** offset is exercised
  on device for the first time — it is `srcOff[1]`, which the kernel receives
  as **`o2`** (the kernel is called `PadCopyKernel(..., o1=srcOff[0],
  o2=srcOff[1], o3=srcOff[2])`, so `o3` is 0 here; **the literal `o3`
  parameter, non-zero only for a rank-5 three-spatial-dim conv, remains
  untested** — recorded as the one residue, and the coverage claim is stated
  as the width axis, not as `o3`)
  (an offset-0 copy of the same destination gives
  10.5 and 2.125 against the oracle's 0.75 and −1.25, max diff 9.75, so the case
  discriminates); and `RejectsNegativeEndPadding` uses `pads=[1,0,0,-1]` →
  `pad_end=[0,-1]`, exercising the guard's **negative-end-pad** arm, which had
  never been tested, with the **full message including the node-name context**
  asserted (`Conv (node 'cv_negpad_end'): negative padding is not supported on
  CUDA here`). The pre-existing `RejectsNegativePadding` is unchanged. The
  guard's two arms and both spatial axes are now covered, and the diff against
  HEAD is additions only.
- **The Grounding DINO OpenCL GAP is now MEASURED on the final tree** (packet
  GD-OCL, one run, no retry): exit **1**, OpenCL status **−4**
  (`CL_MEM_OBJECT_ALLOCATION_FAILURE`), the **same node** as the pre-GD-2c
  record — `Add` at
  `/model/model/encoder/layers.0/fusion_layer/attn/values_vision_proj/Add` —
  **no outputs written**, wall **95.74 s** (pre-GD-2c 98.88 s), max RSS
  **2,866,116 kB** (2,865,924 kB), post-run idle **16 MiB with zero compute
  apps** across 24 consecutive samples — no leak — and the sampler is verified
  stopped by `ps`. The tree was confirmed to be the **handle path**
  (`OpenCLDeviceImpl::supports_arena_offsets()` → `false`) with the binary
  post-dating the archive. **One mismatch, reported as a finding:** the peak
  device memory is **5,510 MiB** (board 5,531) against the pre-GD-2c record's
  **5,364 MiB** — but 5,510 MiB is *identical* to the GD-2b re-run's figure, so
  the pre-GD-2c number is the outlier of three samples of the same failure and
  the high-water mark is timing-dependent under the driver's lazy migration of an
  11 GiB construction demand. The **execution index is not reported by this
  harness**, so that line is *not comparable*; the pre-GD-2c figure (exec
  14,394 / ONNX file index 23,997) comes from the GD-2b plan probe and is
  cited as such, not re-derived.
  **The numbers the architect required for the OpenCL cell, now measured:** the
  reverted handle path's executed demand is **1,303 buffers / 10,101.0 MiB** of
  slots, **11,028.5 MiB** of total construction demand against a **5,803.3
  MiB** device (**1.90x**), while the model's own **live set is 1,687.9 MiB —
  29 % of the device**. So the −4 is a device-capacity outcome of the handle
  path's **over-allocation**, not of the model's requirement, which is exactly
  why the cell reads **backend GAP** and not "environment-limited".
- **The Grounding DINO CPU leg is measured on the final tree.** The re-run
  (session `ses_eaa4499beffeYDFAJIYsa4gxBC`) gives `GD4A2-LEG: EXIT-0`,
  **19,224** nodes, wall 1:16:39, RSS 2,865,992 KB, and the same raw figures as
  the first attempt (logits max 1.606933 / mean 1.323906e-02 / **47,138** of
  230,400 over `1e-3(1+|ref|)`, max ratio 315.6493; pred_boxes **72** of 3,600,
  max ratio 963.7469; zero non-finites). **Both output tensors are
  byte-identical to the first attempt** (sha256 `c72245f5…` and `4ee88d48…`,
  `cmp` exit 0), as is the YOLO26n CPU control (`ff9b1f1f…`), so the GD-2c
  revert is **measured** to be CPU-arithmetic-neutral rather than inferred.
  **The acceptance command exited 1, and the sole failing clause was a proxy the
  packet text specified wrongly:** `test -z "$(git diff HEAD -- src include test
  …)"` cannot hold while the accepted RTV4 CUDA remedy sits legitimately
  uncommitted in `src/` and `test/` pending the same-commit rule. The packet's
  *intent* — wrote no tracked repo file — is verified (its only repo-directory
  writes were git-ignored `build/` outputs; all 11 GD-2c-era files are
  byte-identical to HEAD), so the failure is a mis-specified proxy, not a
  measurement defect, and a re-run would produce the same red. The closing
  green acceptance for the tree is the orchestrator's boundary run, after the
  commit. Two path facts for the next packet: the YOLO26n spot-check input is
  **not** `/tmp/opencode/yolo26n_input.bin` (which does not exist) but the
  runner's hard-coded
  `$NEURIPLO_REPO/build-parity/native_yolo_parity_fixture/yolo26n_input.bin`,
  the same file the first attempt used; and **`/tmp/opencode/yolo_gate_runner`
  is now the CPU-archive build** (it needed a throw-only `nert::OpenCLDevice()`
  stub, because the CPU archive does not export it), so any packet needing the
  OpenCL path must relink it against `build-ocl/libnert.a`.
- **The Grounding DINO CPU leg is reviewed: the measurement is accepted, the
  acceptance command is not a valid gate** (reviewer session
  `ses_ea9d2d655ffeTHRkY69DJ17mfc`, verdict PASS-WITH-FINDINGS). The reviewer
  independently recomputed every compare number (logits max 1.606933 / mean
  1.323906e-02 / **47,138** of 230,400 over `1e-3(1+|ref|)`, max ratio
  315.6493; pred_boxes **72** of 3,600, max ratio 963.7469; `ENV-CPU-RAW`
  max 963.746903 / mean 2.114396 / 47,210 of 234,000 over ratio 1, the
  definition stated in the script rather than assumed), confirmed the
  byte-identity of both tensors and the spot-check pair, verified the archive
  by hashing all **81** members against the current objects (0 mismatches, 0
  missing) with `NERT_ENABLE_CUDA:BOOL=OFF` and no `cuda|opencl` symbols in
  the archive or the binary, and confirmed the eleven GD-2c-era files are at
  HEAD. **The literal acceptance is rejected as a closing gate**, for three
  reasons: (a) it cannot pass while the RTV4 remedy is legitimately
  uncommitted; (b) it greps `CODE-TREE-CLEAN: YES`, a **worker-authored,
  self-attested** line that is false as a statement about the tree; and (c) it
  is **cwd-fragile** — from outside the repository `git diff` yields empty
  stdout and the clause **false-passes**. The corrected closing acceptance,
  which encodes the actual intent ("this packet wrote no tracked CPU-relevant
  file"), is:

  ```
  test -f /tmp/opencode/gdino/gdfin2_cpu_logits.bin && \
  test -f /tmp/opencode/gdino/gdfin2_cpu_pred_boxes.bin && \
  grep -q '^GD4A2-LEG: EXIT-0' /tmp/opencode/gdino/gdfin2-cpu-evidence.md && \
  grep -q '^SPOT-CHECK: IDENTICAL' /tmp/opencode/gdino/gdfin2-cpu-evidence.md && \
  grep -q 'nodes_dispatched=19224' /tmp/opencode/gdino/gdfin2_cpu_run.log && \
  grep -q '^NERT_ENABLE_CUDA:BOOL=OFF' $NERT_REPO/build/CMakeCache.txt && \
  git -C $NERT_REPO diff --quiet HEAD -- src include test CMakeLists.txt test/CMakeLists.txt \
    ':(exclude)src/cuda/conv.cu' ':(exclude)src/cuda/conv_frontend.cu' \
    ':(exclude)test/cuda_conv_test.cpp'
  ```

  and `CODE-TREE-CLEAN: YES` must be replaced by the fact it stands for —
  `PACKET-WROTE-TRACKED-FILES: NONE`, plus `EXTERNAL-CODE-DIFF:` naming the
  three RTV4 files as not in the CPU build — with the three excluded files
  pinned by hash so a later packet cannot hide a change in them. **The closing
  acceptance for this packet is the orchestrator's boundary run, after the
  commit**, when the RTV4 remedy is committed and the clause passes naturally.
  Three further corrections the reviewer recorded: the evidence's claim that
  the code diff was "byte-for-byte what it was at dispatch" is **false** (the
  test file went 89 → 263 changed lines during the leg, at 22:53); the repo
  monitor starts at 23:16:07, **after** the leg ended at 23:14:40, so "the
  packet wrote no tracked file" is **inference from mtimes and content, not
  observation**; and the 13-sample tail after 23:45:23 has no artifact. The
  measurement is unaffected by all three — the three changed files are CUDA-only
  and the archive has no CUDA objects — but the *attribution* is now stated
  accurately. Two standing cautions: **byte-identity is not a numerical pass**
  (the comparator still reports 47,210 of 234,000 over the relative bar, and
  the task-equivalence judgment is GD-TG's), and the archive was 77/81
  recompiled, not 81/81 (the four reused objects post-date their sources, so
  this is wording, not staleness). The spot-check **overlapped the leg** by
  ~30 s, which affects the leg's wall/user timing but not its numerics.
  **Runner provenance:** `/tmp/opencode/yolo_gate_runner` is now the
  **CPU-archive** build (it needed a throw-only `nert::OpenCLDevice()` stub
  because the CPU archive does not export it); the OpenCL-archive control is
  `/tmp/opencode/yolo_gate_runner_oclarch`, or relink against
  `build-ocl/libnert.a`. **Residual risk:** the CPU leg is final-tree only for
  the CPU subset — if the eventual commit of the remedy touches a CPU-relevant
  source, a shared header or CMake, the archive changes and the byte-identity
  no longer applies, so the CPU leg must be re-run. Today it touches only
  `src/cuda/*` and `test/cuda_conv_test.cpp`, both outside the CPU archive.
- **The Grounding DINO task gate PASSES on both devices** (packet GD-TG,
  session `ses_ea9d3289dffelOxcO0E5YkbCHg`, acceptance exit 0,
  `GD-TG-ACCEPT`). The TopK permutation was **derived from the fresh outputs**,
  never hard-coded: CPU **9** swaps (18 of 900 slots), CUDA **10** (20 of 900,
  including 785/786). The derivation was cross-checked against an exact DP
  optimum, an independent logit-domain audit with no box data, and an
  independent box-identity check, and it **reproduces the calibration's
  structure as a result** (the fresh canonicalized mean 7.639468e-03 against the
  calibration's 7.639467e-03, 7 s.f.); one dissent is recorded — an
  unstandardised Chebyshev metric gives CUDA 9, missing 785/786 because a
  near-duplicate `ref[707]` shadows it.
  **Criteria, raw and canonicalized:**

  | criterion | CPU raw | CPU canonicalized | CUDA raw | CUDA canonicalized |
  |---|---|---|---|---|
  | (a) `pred_boxes` over `1e-3(1+\|ref\|)` | 72/3,600, max 9.744847e-01 | **0/3,600, max 7.003546e-05** | 80/3,600, max 9.744847e-01 | **0/3,600, max 7.027388e-05** |
  | (b) per-box IoU | min 0.0, 882/900 | **min 0.9997353, 900/900** | min 0.0, 880/900 | **min 0.9997261, 900/900** |
  | (c) top-1 token | **896/900** | **900/900** | **896/900** | **900/900** |
  | (d) decoded set | kept 1, set_diff 0 | kept 1, **set_diff 0**, token 1/1 | kept 1, set_diff 0 | kept 1, **set_diff 0**, token 1/1 |
  | (e) envelope | max 1.606933, mean 1.323906e-02 | max 4.459157e-01, mean 7.350136e-03 | max 1.607923, mean 1.373600e-02 | max 4.419680e-01, mean **7.639468e-03 — BREACH 1.86 %** |

  Criterion (c) is defined on the **canonicalized** count: raw counting would
  score 896/900 and fail **for the wrong reason**, because all four raw
  mismatches (697, 698, 843, 844) are TopK near-tie reorderings inside swapped
  slots, not numeric deviations. The CUDA envelope breach is **surfaced, not
  smoothed**, and is **not** treated as a device FAIL — it is the envelope's
  per-device recalibration trigger, and the raw bounds are met.
  **The decode margin is decisive:** one kept query (index 0) on both devices,
  borderline score 0.2639097 against a margin of **1.390966e-02**, canonicalized
  max \|Δ max-score\| 5.337870e-03 (CPU) / 5.362389e-03 (CUDA), kept-query
  score delta 3.309111e-05 / 2.294870e-05, so max-delta − margin is
  **−8.571794e-03 / −8.547275e-03**: no query can cross the threshold at any
  setting. The safe interval for the kept set is (0.1253985, 0.2638766) CPU and
  (0.1253994, 0.2638867) CUDA. The logit-space top-1 delta (2.168975e-01 /
  2.169518e-01) is a **different quantity** and is reported only for contrast.
  The threshold provenance is recorded verbatim: the Neuriplo repository contains
  **no** production threshold (`NertInfer.cpp` is 235 lines and threshold-free),
  so 0.25/0.25 is the **Hugging Face proxy**, and the residue is that a
  consumer using different thresholds could see different kept sets. OpenCL is
  excluded and not blended: the backend GAP's measured facts are cited from
  `gdocl-evidence.md` (exit 1, status −4, no outputs, 11,028.5 MiB demand
  against a 5,803.3 MiB device, 1,687.9 MiB live set = 29.1 %).
  A measures-versus-cites table separates the 13 measured items from the 7 cited
  ones, with the calibration pair list and the raw/aligned figures marked
  **calibration only**.
- **The GD-6 integration was reviewed and corrected before any commit**
  (reviewer session `ses_ea9b2b5d8ffe9Yf6Y8C443gxe3`, verdict
  PASS-WITH-FINDINGS). The seg-nano Evidence cell was confirmed
  **byte-for-byte** identical to the architect's approved text (1,057 characters,
  compared against the ledger's line, not the worker's report), and the
  scoreboard arithmetic and membership were confirmed (15 = 5 primary + 8
  variants + RT-DETRv4-s + seg-nano-576, with Grounding DINO named only under
  Open). Six defects were corrected in `coverage.md` and `CHANGELOG.md`:
  (i) the Grounding DINO row carried **unescaped pipes** in `1e-3(1+|ref|)`,
  giving 9 pipe bytes against the table's 7 and corrupting the rendered Evidence
  column — now `1e-3(1+\|ref\|)`; (ii) the Grounding DINO cell presented
  post-canonicalization numbers without saying so and omitted the raw ones —
  the same omission class that voided the architect's grant on the seg-nano
  cell — so the cell now states that **gate metrics are post-canonicalization**
  and carries the raw counts (pred_boxes **72/3,600** CPU and **80/3,600** CUDA
  at max abs 9.74e-01; per-box IoU min 0.0 with 882/900 and 880/900 above
  0.999; and the **raw** max |Δ max-score| 2.12e-02, which **exceeds** the
  1.39e-02 margin, so "no query can cross" holds canonicalized only); (iii) the
  `### Fixed` int64 `Clip` bullet had a stray `(` and a dropped final period;
  (iv) a stale `### Added` sentence still claimed the CUDA run, the ORT
  comparison and the OpenCL leg were "still pending"; (v) the closed
  RT-DETRv4-s item was still listed under "Open items" (now a single prose line
  names the three closed items, and the remaining list holds the OpenCL gap and
  the missing exports); (vi) the matrix date was 2026-10-19. Plus a naming
  alignment (the scoreboard's "(TopK deviation)" → "(SN deviation)").
  **The correction packet's own acceptance command failed on a defect in the
  command, not in the work**: `grep "^\| $r "` is parsed as a BRE alternation
  (`^` OR ` Grounding DINO `), so it matched every line and returned the
  whole-file pipe count. The command also carried a **latent contradiction I
  wrote in**: the escaped `\|ref\|` contributes two extra pipe *bytes*, so
  "exactly 7 pipe bytes" and "contains the escaped form" cannot both hold.
  The **correct** check counts **unescaped** pipes
  (`re.findall(r'(?<!\\)\|', row)`), and the orchestrator's own corrected
  acceptance over the final files is green: all three rewritten rows have **7
  table delimiters** (the Grounding DINO row has 9 raw pipe bytes = 7
  delimiters + 2 escaped), the architect's seg-nano cell is **byte-identical**
  to the approved text, the raw Grounding DINO counts, the escaped tolerance,
  the date and the scoreboard wording are present, the changelog's stale
  sentence is gone (0 matches) with the `Clip` bullet repaired, and the scoped
  file set is exactly `coverage.md`, `CHANGELOG.md`, `README.md`. **Lesson for
  the program: an acceptance command that counts a delimiter must say whether it
  counts escaped occurrences.**
- **Numbering note.** The architect's line says "closes Open Work #3"; in the
  maintainer's slim ledger the seg-nano item is **#2** and the RT-DETRv4 item
  is **#3**. Both close at this integration — #2 by the SN-DEV ruling and #3 by
  the already-measured RT-DETRv4-s CUDA gate — so the reconciliation is
  recorded here rather than left to a reader of the quoted line.
- **Same-commit rule.** The PASS 15 requires the **uncommitted** asymmetric-pad
  remedy (`src/cuda/conv.cu`, `src/cuda/conv_frontend.cu`,
  `test/cuda_conv_test.cpp`) to be committed in the same integration as the
  documentation, because RT-DETRv4-s's CUDA PASS rests on it. A
  documentation-only integration must leave that row at `PASS | GAP | PASS`
  and record **14 / 2 / 14 / 2**. Seg-nano itself is unaffected either way —
  its CUDA outputs are byte-identical to the pre-fix artifact.

## Prepared decomposition — the export-first provisioning batch (next session)

Planned by the planner (session `ses_eaa1493b7ffeVWftPmPbOZxWyN`) for the
**next** orchestrator session, because it is the last major workstream before
0.2.0 can close and it shares no files with the closing integration. The
Neuriplo repository (`$NEURIPLO_TASKS_REPO`, `NT/`) is **read-only** —
running an exporter that writes into `/tmp` is fine; editing `NT/` is not.
Toolchain: `transformers 4.35.0`, `torch 2.12`, `onnx 1.22`, `onnxruntime 1.27`,
`ultralytics 8.4.113`; `import rfdetr` **fails** (`BackboneConfigMixin` needs
transformers 5.x). NERT loads **exactly opset 18**
(`src/model_loader.cpp:109-111`).

### Feasibility — all 14 missing-export groups

| # | Group | Exporter | Opset → 18 | Checkpoint | Verdict |
|---|---|---|---|---|---|
| 17 | YOLO detector v5–v12 | `NT/export/detection/yolo/export.py:281`, `--opset` at `:690` (default 12) | pinned at export | downloadable | with download |
| 18 | YOLO segmentation | **no `.py`** — `NT/export/segmentation/InstanceSegmentation.md:1-12` (Ultralytics CLI) | pinned at export | **local** `$LOCAL_ARTIFACTS/tritonic/yolo11m-seg.pt` | **offline** |
| 19–21 | EdgeCrafter det/seg/pose | **no exporter in repo** — `NT/export/**/edgecrafter/README.md:1-30` defers to upstream | convert if needed | absent (GitHub release) | with download (heaviest setup) |
| 22–24 | RF-DETR det/seg/keypoint other sizes | `NT/export/**/rfdetr/export*.py` (default 17) | pinned at export | absent **and the dependency is broken** | **not executable here** |
| 25 | OWL-ViT | **no exporter in repo** — a new worker-written script under `/tmp` | pinned at export | downloadable (HF) | with download + new exporter code |
| 26 | ViTPose-base-simple | `NT/export/pose_estimation/vitpose/export_vitpose_to_onnx.py:137-149` (default 14) | pinned at export | downloadable | with download |
| 27 | Depth-Anything-V2 | `NT/export/depth_estimation/depth_anything_v2/export_depth_anything_v2_to_onnx.py:138-152` (default 14) | pinned at export | downloadable | with download |
| 29 | VideoMAE / ViViT / TimeSformer | three exporters, all default 14 | pinned at export | downloadable | with download |
| 30 | Classification (incl. ResNet-50) | `NT/export/classification/torchvision/export_torchvision_classifier.py:38-46` — **no opset argument at all**; `.../vit/onnx/deploy.py:15,332` (default 17) | verify, then convert if needed | downloadable | with download |
| 31 | RAFT-small | `NT/export/optical_flow/raft/raft_exporter.py:174` **hardcodes opset 16** | **converted** to 18 | downloadable | with download + conversion |

**Totals: 1 offline, 10 with download, 3 not executable here** (the RF-DETR
groups, blocked on a transformers ≥5.x upgrade and on non-hermetic weight
downloads from `storage.googleapis.com`).

### The reusable method — five steps, identical per group

1. **PROVISION** — one opset-18 ONNX at `/tmp/opencode/exports/<group>/`,
   checker-clean, with a `provenance.md` (command, sha256, node count, declared
   inputs/outputs) and a `verify_export.py` whose **acceptance asserts
   `opset == 18`, the checker passes, nodes > 0 and non-empty io**. If the
   export was *converted* rather than pinned, the provenance must carry the
   verbatim caveat that **conversion does not prove semantic equivalence** — the
   full-model gate is the equivalence evidence.
2. **PROBE** — load it through NERT's own loader and shape inference; run
   `nert_gate_cpu` on zero-filled inputs shaped from the graph's declared dims
   as the **load proof**; record `nodes_dispatched=` (post-constant-folding, the
   census convention), the **op-type histogram** over the on-disk graph, and
   the **missing-row diff** per device table (census minus the registered
   symbols in `src/kernels/cpu_kernels.cpp`, `src/cuda/table.cu`,
   `src/opencl/table.cpp`). The verdict is GO (all three empty) or BLOCKED
   (listing the rows). The current OpenCL table lacks `Abs`, `Floor`,
   `GreaterOrEqual`, `Neg` — none used by a currently loadable entry, so a new
   export may well need them.
3. **GATE** — a deterministic fixture (the `bus.jpg` pattern in
   `/tmp/opencode/catalog_refs/`; 640×640 detection/seg/pose/depth, 224×224
   classification, a static 5D fixture for video, an image pair for RAFT, image
   plus tokenised text for open-vocab), the **ORT reference** for every declared
   output, the **three device legs** with **equal `nodes_dispatched`** across
   devices, `gate_compare.py --tol 1e-3` per device, and the task analysis
   (`--top1` for logits, `--iou` for masks, the per-query IoU idiom for
   segmentation). The evidence file follows the standing `*_evidence.md` format.
   **The acceptance asserts each leg's exit status == 0 and the node-count
   equality** — `gate_compare.py` alone cannot gate, because a byte-identical
   output written before a crash is not a pass.
4. **REVIEW** — a fresh read-only reviewer per implementer packet, inspecting the
   real artifacts, the writable-path list and the acceptance evidence; a packet
   is accepted only with reviewer PASS **and** green acceptance.
5. **INTEGRATE** — one serialized owner writes the coverage row (and a
   `PASS (…deviation)` cell with the full deviation text where a ruling
   requires it), the scoreboard counts and the changelog entry. Never parallel
   with another integration; the orchestrator commits the scoped paths and pushes.

### Tranche 1 — the four highest-leverage groups

`E2-PROV-yolo11m-seg` runs **first** (offline, ~10 min, no network), then
`E1-PROV-yolov8n`, `E3-PROV-owlvit` and `E4-PROV-resnet50` in parallel tracks
under the two-packet cap, each followed by its PROBE, then its GATE — and
**only one GATE may hold the GPU at a time**. Writables are disjoint
(`/tmp/opencode/exports/<group>/`, `/tmp/opencode/catalog_refs/<model>/`,
`/tmp/opencode/gates/<model>_*`). A separate `INT-1` integration packet covers
whichever tranche-1 gates passed with reviewer PASS. Per-group detail — exact
exporter commands, declared input/output names, expected folded counts, fixture
shapes, and the `verify_export.py` / `probe_accept.sh` / `gate_accept.sh`
acceptance commands — is in the planner's output for session
`ses_eaa1493b7ffeVWftPmPbOZxWyN`; the tables above and the method are the
dispatch-ready part.

### The kernel-row rule, stated honestly

**An export packet never adds a kernel row.** If a probe's `missing_rows.txt` is
non-empty, the probe verdict is BLOCKED and the orchestrator files a **separate
kernel packet** (writable `src/opencl/table.cpp` plus the row's kernel source
and tests), with its own device-parity decision, test, review, and — because a
kernel row changes shared engine code — a re-run of the affected existing gate
legs that must show byte-identical outputs before its own acceptance. Sequence:
PROBE-BLOCKED → kernel packet (implement → review → accept) → re-run the probe
acceptance → GATE. Tranche 1 expects E1/E2/E4 GO with no kernel work (the same
op families as gated precedents); E3 (OWL-ViT) is the one candidate that could
surface a missing row, and it is budgeted as one conditional kernel packet.

### Release-gate arithmetic

After the pending integration lands (with the same-commit rule — the
asymmetric-pad remedy and the documentation rows together, else seg-nano-only
is 14/2/14/2): **PASS 15 / Open 1 (Grounding DINO) / Missing export 14 / Out of
scope 2**, 15 + 1 + 14 = 30 in scope. The gate is **zero open entries and zero
missing exports** among the 30; HIP is outside it. Tranche 1 delivers 4 of the
14 missing-export entries (17, 18, 25, 30). Extra sizes beyond one
representative per entry **do not move the scoreboard** and are scheduled after
the gate.

### Maintainer decisions this batch needs (not worker calls)

1. **The RF-DETR groups (22–24): `MISSING-EXPORT` with the reason, or
   `OUT-OF-SCOPE`?** Recommended: `MISSING-EXPORT (rfdetr import blocked under
   transformers 4.35; weights non-hermetic)` — the factory accepts these names
   and other RF-DETR entries are PASS, so out-of-scope would misdescribe the
   catalog. The alternative is a transformers ≥5.x upgrade plus a
   network-allowance ruling, an environment decision with regression risk to the
   15 PASS entries.
2. **One tranche or one group at a time?** Recommended: tranche 1 as four
   parallel tracks under the two-packet cap, GPU serialized.
3. **Does one gated representative per entry close that entry?** Recommended:
   yes. If every size must gate, the scoreboard denominator needs redefining
   before dispatch.
4. **Grounding DINO's OpenCL GAP against the release gate.** The GAP is a
   backend limit on this driver, not an export problem, and **no export-first
   work unblocks it**. Either the gate admits a documented GAP-with-deviation
   for Grounding DINO, or 0.2.0 waits on an OpenCL allocation-strategy change
   (which would need a driver-level trace, per the standing rule — a modelled
   footprint does not qualify).
5. **The conversion evidence bar.** Converted exports (RAFT-small, possibly
   ResNet-50 and EdgeCrafter) carry "conversion does not prove semantic
   equivalence"; the full-model gate against ORT is the equivalence evidence.
   If a stronger bar is wanted (an ORT-vs-ORT pre/post-conversion diff), it
   must be added to the GATE template **before** tranche 1 runs.

### Integration prohibitions from the SN-1 / SN-2 review (session `ses_ea9d27a3fffeyVPbTkh3nlFUX5`)

Both packets: **PASS-WITH-FINDINGS**; every decisive number reproduces exactly,
including an **independent in-memory ORT 1.27.0 CPU re-run** that confirmed the
triple-zero neutrality and the reference bit-identity. The selection rule was
re-derived by applying the engine's own rule (descending value, ascending index
on exact ties) to all four score vectors: **100/100 match for ORT, CPU, CUDA and
OpenCL** — and there are no exact ties in any top-101, so the rule is verified
for this model, not merely asserted.

**The highest-risk item in the program is an integration wording risk, not a
measurement one. The integration packet must:**

1. **Copy the seg-nano Evidence cell from this ledger's architect-ruling
   section verbatim.** It must **NOT** use `sn2_evidence.md` §6's "proposed
   deviation wording" — that is the **thin version the architect rejected**
   (it omits the pinned basis, the unpinned over-tolerance counts and the CUDA
   0.984375), and using it **voids the grant**, reverting the row to
   `PASS | FAIL | FAIL`.
2. **Not** write a bare `PASS` on CUDA or OpenCL, or any wording implying
   unpinned output equality or bit-identity.
3. **Not** write "0 over tolerance" for the GPU masks: it is **2 / 36 / 17** of
   2,073,600 pinned (max ratios 1.27923 / 2.50581 / 1.25271) and
   **1,240,141 / 1,239,727** unpinned (max ratios 47,350.3 / 47,350.4).
4. **Not** write "CUDA IoU 1.0 at raw≥0.5": it is **0.984375** (query 36, inter
   63 / union 64); 1.0 is CPU and OpenCL at raw≥0.5, and **0.999800 is the
   raw≥0 minimum on all three** (query 60, inter 5003 / union 5004).
5. **Not** describe the CPU as the clean oracle, as bit-identical, as quieter, or
   as having the largest deviation. It has 2 marginal mask elements over
   tolerance, its max score-vector deviation is 65.7x the reference margin, and
   its own max noise is 39.4x its own margin.
6. **Not** imply the pinned graph is deployable or generally equivalent: the
   neutrality is **single-input** and the counterfactual is for this gate. And
   **not** present the pinned PASS as exercising NERT's TopK kernel — the pinned
   graph has 1,215 dispatchable nodes against the original's 1,216, with the
   TopK computation removed. The *unpinned* SN-1 signature is what identifies
   the TopK selection as the failure mechanism.
7. **Not** claim `PASS 15` from seg-nano alone: that needs the RT-DETRv4-s CUDA
   cell **and** the asymmetric-pad remedy committed in the same integration
   (otherwise `14 / 2 / 14 / 2`).
8. **Not** follow the older `seg_nano_evidence.md` §5.2 recommendation of an
   engine-side epsilon or near-tie tie-break. The ruling forbids it — the GPU
   values are strictly ordered, so a band of at least 3.58e-06 would be
   fabricated — and it would require a separate ruling and a full re-gate. The
   tree has no TopK diff, so the constraint is presently respected. **Do not
   modify `src/cuda/top_k.cu`, `src/opencl/index_move.cpp` or
   `src/kernels/top_k.cpp`.**
9. **Not** describe "max ORT−device deviation" as an output deviation: those
   figures are maxima over the **score vector** `/transformer/ReduceMax_output_0`
   (2,304 elements), not over `dets`/`labels`/`masks` (where the CUDA masks
   max abs is 70.98). The **corrected cell** now says "score-vector" and names
   the tensor and its length, and it labels the unpinned pairs **inside** the
   cell — so this prohibition is discharged by the cell itself, and the
   instruction to "label it outside the verbatim cell" is **withdrawn**
   (architect clarification, 2026-10-20).
10. **Not** overclaim the pin's provenance: the fresh derivation is
    **byte-identical** to the pre-existing `seg_nano_topk_pinned.onnx`
    (`db20d63b…`), so "independent reproduction, not a copy" is **unverifiable
    from the artifacts** (the process claim is corroborated — `sn2_pin.py` builds
    from the original and the constants match a different packet's probe
    captures — but the bytes cannot distinguish it). The pin's validity does not
    depend on independence: its structure and the ORT neutrality are verified.
11. Note for the record: the older `seg_nano_evidence.md`'s "90/91" top-1 line
    is a misleading comparator artifact; cite **99/100**.

**One item is referred to the architect for clarification** (the condition says
the row must carry "the unpinned query-99 flip and over-tolerance counts", while
the architect's own verbatim cell carries the unpinned **dets and labels**
counts but not the unpinned **masks** counts). The row is not to be edited
silently; the clarification is requested.

### Architect clarification on the SN-DEV coverage cell (2026-10-20)

**The earlier "verbatim" cell was superseded.** The architect ruled that the
**condition is the operative grant** and the "copy this verbatim" instruction
was only the mechanism for satisfying it — so the cell has been **corrected in
place** (ledger line 281) and the earlier text must not be used.

**Why the omission was material, not cosmetic.** On the old cell a reader saw
unpinned `dets 50/400` and `labels 3,629/3,628` against *pinned* `masks
2/36/17`, and could only conclude the unpinned GPU failure was a
detection-level affair. In fact **1,240,141 (CUDA) and 1,239,727 (OpenCL) of
2,073,600 mask elements — ≈59.8 % — were over tolerance unpinned**, at max
ratios of **47,350.3 / 47,350.4**. A row that omits that understates the
deviation while claiming `PASS (SN deviation)`, which is exactly the "implication
of unpinned output equality" the condition voids on. The corrected cell carries
the unpinned masks counts and ratios, labels the unpinned pairs by device
**inside** the cell, names the score-vector tensor and its length, and states
that the pinned graph is a **single-input counterfactual for this gate (TopK
removed; 1,215 vs 1,216 nodes): not a TopK-correctness result, not a deployable
substitute**. The changes are **additive only** — no clause, number or
qualifier was removed, so the grant is not widened.

**Rule the architect set for the two artifacts:** the **cell is the published
record** of the condition; the **ledger is the working basis**. The unpinned
masks counts belong **in the cell** (required) and must not be left to a
cross-reference from `coverage.md` to this ledger. The architect chose the
explicit clarification over parking the counts in the ledger's basis paragraph,
because `coverage.md` is the artifact that outlives the ledger for a reader of
the matrix.

**Other conditions the integration must honour** (architect, 2026-10-20):
- `sn2_evidence.md` §6 **remains forbidden** — it is still the thin wording
  that omits the pinned basis, the unpinned counts and the 0.984375, and using
  it voids the grant.
- The **CPU cell stays an unqualified `PASS`** and may not be described as
  bit-identical, as an oracle, or as quieter — it has 2 mask elements over
  tolerance.
- The scoreboard is `PASS 15 / Open 1 / Missing 14 / Out of scope 2` **only if**
  the RT-DETRv4-s CUDA cell and its asymmetric-pad remedy land in the same
  integration; seg-nano alone is `14 / 2 / 14 / 2`.
- **No engine change**: `src/cuda/top_k.cu`, `src/opencl/index_move.cpp`,
  `src/kernels/top_k.cpp` stay untouched; any epsilon or tie-break needs a new
  ruling and a full re-gate.
- Revisit triggers unchanged, including a `raw>=0.5` release bar of
  `IoU >= 0.999` (CUDA is 0.984375) and any change to the export, the
  reference, the tolerance or the TopK order.
- **Do not "improve" the ratios**: `47,350.3 / 47,350.4` exactly as recorded —
  they must not be recomputed from a rounded tolerance or restated as max abs.

**One factual slip in the clarification's prose, recorded so the record is
accurate.** The architect's condition list says the CPU has "the largest
score-vector deviation (9.39e-05)". It does not: **OpenCL's is larger**
(1.24216e-04 against the CPU's 9.39369e-05, with CUDA smallest at
4.81606e-05) — and the architect's own corrected cell states OpenCL's figure
correctly while making no claim about the CPU's rank. The facts the integration
must state are: the CPU's max score-vector deviation is **65.7x the reference
margin** and larger than CUDA's, though smaller than OpenCL's; the CPU's own
margin is **+2.38419e-06**; and the CPU's own max noise is **39.4x its own
margin**. The prohibition on presenting the CPU as the clean oracle stands
whatever the ranking — it is the architect's rule and it is correct.

### Review of the residual-coverage packet (session `ses_ea9ca227affebvp5Y6Uwur5bck`)

**PASS-WITH-FINDINGS** — all three residuals are closed **on substance**, the
test file is committable with the remedy, and the diff is additions-only. The
reviewer independently reproduced the CPU oracle in Python and confirmed the
discrimination: for `x(1,2,3,3)`, `w(2,2,2,2)`, `b={0.75,−1.25}`,
`pads=[0,1,0,0]` the oracle gives `m0 row0 [0.75, 10.5, 5.25]` and
`m1 row0 [−1.25, 2.125, 6.75]`, while an offset-0 copy gives
`m0 row0 [10.5, 5.25, 1.125]` and `m1 row0 [2.125, 6.75, 2.875]` — max
difference **9.75** against a tolerance of 0.0115 at the largest reference, so
`ExpectNear` fails on the bug. The negative-end case can only trip the
`pad_end < 0` arm: `pad_begin` is non-negative, the output dims match the
formula (`{1,1,3,1}`), and the guard precedes the shape check — so with that arm
removed the row would proceed to cuDNN and `EXPECT_THROW` would fail.

**The one defect was in this ledger, not the code:** the test comment, the
evidence and the ledger all called the width offset **`o3`**, but for the 4-D
geometry the kernel receives `srcOff[1]` as **`o2`** and `o3` is 0
(`src/cuda/conv.cu:165-166`, `:377-378`). The width-axis coverage and the 9.75
discrimination are real, so the residual is closed, but the record overstated
`o3`. Corrected above, together with the test name (the actual test is
`AsymmetricWidthBeginPadWithBias`, not `…Mixed…`). **The literal `o3`
parameter — non-zero only for a rank-5 three-spatial-dim conv — remains
untested**, and is recorded as the one residue.

Three further notes for the integration: the **12.70 s** summary figure is
console-only (the final `LastTest.log`'s per-test times sum to **12.66 s**, and
the only saved console log is a 22:55:07 development run) — cite the final log
for 24/24 and the 14-test block, and 12.66 s if a time is needed; the 89
RTV4-1 lines in the test file are **byte-identical** to the saved accepted diff,
while the 85 follow-up lines are additions-only and in place but have no
archived snapshot to prove byte identity; and `build-cudnn`'s
`LastTestsFailed.log` (20:08:02) is **stale**, predating both the remedy gate
and the final run, whose `LastTest.log` shows `nert_cuda_conv` passing.

**Expected diff shape for the commit:** `test/cuda_conv_test.cpp` → **+263 / −0**,
one insertion hunk `@@ -210,0 +211,263 @@`, 322 → 585 lines — the 263 are 89
(RTV4-1) + 85 (test-gap follow-up) + 89 (this packet), because the first 174 are
still uncommitted on top of HEAD. `src/cuda/conv.cu` and
`src/cuda/conv_frontend.cu` are **+115 / −2** each and byte-identical to the
accepted remedy diff. The `nert_cuda_conv` ctest entry has **no gtest filter**
and runs the whole binary (14 tests).

### Reconciliation with `origin/main` (2026-10-20)

The push of this boundary's first commit was **rejected**: the maintainer had
pushed `604ee3e` ("docs: remove local home paths from specs") at 00:18 from a
**stale** copy of `inventory.md` — the 354-line pre-slim census — while the
slimmed 105-line census (21:19, carrying the `$HERMES_CACHE` pointer to the
pre-slim archive) had only ever been in their working tree. Reconciled by
rebasing onto `origin/main` and resolving `inventory.md` in favour of the
**slimmed** form: it is the maintainer's working-tree state, it already contains
**no** local home paths (so it satisfies the intent of `604ee3e`), it carries
the archive pointer, and it holds the corrected 15/1/14/2 scoreboard. The
ledger and this boundary's handoff were also stripped of their last local home
paths, and the handoff's legend now defines `$HERMES_CACHE` alongside
`$NERT_REPO`, `$LOCAL_ARTIFACTS` and `${HOME}` — the review had noted that the
pre-slim archive was unreachable from the handoff as written.

### The boundary-reset path is verified (2026-10-20)

The ledger's harness note records that the `orchestrator` agent cannot run as a
subagent, so a boundary reset must go through the CLI. **Both halves of that
path were tested before the boundary, so there is no blocker to report:**
- `opencode api session.create -d '{"title":"…"}'` → returned
  `ses_ea9b3e52bffeopJrPCFf9HcKPe` (a probe session, left idle);
- `opencode run --agent orchestrator --session <id> --auto "<prompt>"` → the
  orchestrator agent started and answered (`READY`) on that probe session.

The reset for boundary 2 will create a fresh session the same way, with the
boundary-2 handoff as the prompt, and the new session id recorded here. The
orchestrator agent's model (`meta/muse-spark-1.3-contributor`) has been serving
without the balance failures that affected the `zai` provider, which is why the
implementer and reviewer agents were re-pointed earlier today.

**A note on the maintainer's concurrent edits, for the commit scope.** The
maintainer has been editing the spec documents throughout this session: the
slimming of `coverage.md` and `inventory.md` (21:19), the README status
sentence (21:23), the path parameterisation in `handoff-boundary-1.md`
(the repository root → `$NERT_REPO`, the local artifacts root →
`$LOCAL_ARTIFACTS/…`) and in `2026-10-18-closure-cuda/validation.md`
(the user's library path → `${HOME}/.local/…`), and the ledger's own slimming.
These are **settled, coherent edits** — the coverage slimming is what the new
rows are written into, so the integration cannot land without them. The
boundary-2 handoff therefore uses the **same placeholder convention**, with a
one-line legend so a reader can resolve `$NERT_REPO` and `$LOCAL_ARTIFACTS`
without hunting for a key.

## Boundary 3 — the export-first provisioning batch (session `ses_ea9a8c0e2ffesxGThz1IUdG8Qf`)

Opened 2026-10-20 from `8215a43` (boundary-2 handoff), branch `main`, tree
clean and in sync with `origin/main`. **The session id was already recorded
in `## Sessions and resets`.**

### Five maintainer decisions — still open, and still the maintainer's

The `question` tool is unavailable in this non-interactive session, so the five
decisions in the handoff could not be put interactively; they are put to the
maintainer **here**. None is settled by this session. Where the batch could
not proceed without one, the working assumption is named below **and is
recorded as open in the published documents**, so a contrary ruling is visible
rather than silent.

| # | Decision | Working assumption used in this batch | What a contrary ruling changes |
|---|---|---|---|
| 1 | RF-DETR groups 22–24: `MISSING-EXPORT` with a reason, or `OUT-OF-SCOPE`? | `MISSING-EXPORT` with the reason (the plan-of-record's recommendation). Not yet written to the census — the three groups remain among the ten open. | The census rows 22–24 and the in-scope denominator. |
| 2 | One tranche, or one group at a time? | One tranche, four groups, **at most two packets in flight**, the GPU single-owner. | Only the schedule. The four gates are done either way. |
| 3 | Does one gated representative per entry close that entry? | **Yes** — the plan-of-record's working assumption, and the architect applied it to entry 25. **Recorded in both the matrix and the census**, so the scoreboard's dependence on it is visible. | The scoreboard denominator: the four closed groups would reopen and the denominator would need redefining. |
| 4 | Does the release gate admit a documented GAP-with-deviation for Grounding DINO's OpenCL cell? | **Not assumed.** The cell stays `GAP`, Grounding DINO stays the one open entry, and **no release was cut**. | Whether 0.2.0 can close once the ten exports are done. |
| 5 | The conversion evidence bar for converted exports. | The prepared GATE template: the **full-model gate against ORT is the equivalence evidence**, with the caveat recorded verbatim in the export's provenance. ResNet-50 (opset 20 → 18) carries it. | A stronger bar (an ORT-vs-ORT pre/post-conversion diff) would have to be added to the GATE template and applied to RAFT-small, and could be applied retroactively to ResNet-50. |

### Tranche 1 — what ran

Five steps per group: **PROVISION → PROBE → GATE → REVIEW → INTEGRATE**, with
a fresh read-only reviewer after every implementer packet. **No kernel row was
added**: the tables are unchanged at CPU 72 / CUDA 72 / OpenCL 68 / HIP 9.

| Packet | Group | Result |
|---|---|---|
| E2-PROV / E1-PROV | YOLO11m-seg / YOLOv8n | opset 18 pinned at export; 515 / 301 nodes on disk |
| E2-PROBE / E1-PROBE | YOLO11m-seg / YOLOv8n | **GO**; 467 / 260 dispatched; all three device diffs empty |
| E2-GATE / E1-GATE | YOLO11m-seg / YOLOv8n | **PASS** on CPU/CUDA/OpenCL; exit 0 on every leg; over_tol 0 on every output |
| E3-PROV | OWL-ViT | opset 18 pinned at export by a **worker-authored** exporter (the Neuriplo repo has none) |
| E3-PROBE | OWL-ViT | **BLOCKED** — the graph did not load; see the architect's ruling below |
| P2 / P2-ACCEPT | OWL-ViT re-export | **12/12** acceptance conditions; E1–E7 evidence met |
| P3 | OWL-ViT re-probe | **GO**; 1,071 dispatched; all three device diffs empty |
| G3 | OWL-ViT gate | **PASS**; exit 0 on every leg; over_tol 0; the ORT-reference byte-identity precondition re-confirmed at gate time |
| E4-PROV / E4-PROBE / E4-GATE | ResNet-50 | opset 18 **converted** from an opset-20 raw export; 122 dispatched; **PASS**, over_tol 0, top-1 1/1 (n = 1) |
| EVID-FIX-1 | — | corrected three factual defects the reviewers found in the evidence/provenance files |
| INT-1 | — | the matrix, census, README and CHANGELOG now carry **PASS 19 / Open 1 / Missing 10 / Out of scope 2** |

### The architect's ruling on the OWL-ViT blocker (E3)

The first OWL-ViT export was **valid but unloadable**: NERT's loader rejected
`/model/class_head/Cast` (`to=11`, DOUBLE — `include/nert/graph.hpp:42` has no
`Float64`) and `/model/Reciprocal` (outside the loader's accepted-op map).
**Both are loader rejections, before any kernel lookup, so kernel rows alone
could not have fixed it.** The architect ruled **re-export, do not touch the
engine**: both blockers are compile-time artifacts of the export, not model
semantics (the DOUBLE is an exact identity round trip; the `Reciprocal` is a
division by a compile-time constant whose whole chain is foldable). A
two-site **exporter-only** patch produced an artifact proven **bit-identical**
to the rejected one on both outputs across five input sets, with the structural
diff confined to the two patched constructs (1,439 → 1,418 nodes, 21 removed,
0 added, 0 violations). The rejected artifact is preserved beside it. **The
engine was not touched**: the loader, executor, shape inference, constant
folding and all three tables are byte-identical to HEAD.

**Four amendments to the program's rules, from that ruling:**

1. **The kernel-row rule.** A missing-row diff taken from a graph the loader
   rejected is **not** a missing-row diff: the graph never reached dispatch.
   `missing_rows.txt` must label each op `loader_rejected` / `loader_folded` /
   `dispatchable_missing`, and **only `dispatchable_missing` can justify a
   kernel packet**.
2. **The tranche-1 budget.** The one conditional kernel packet covers a
   *loaded* graph only; a group whose first provision produced an unloadable
   artifact may be re-provisioned without consuming it (two re-provisions per
   group). E3's kernel budget was returned unspent.
3. **Standing rule 5 — a de-blocking re-export is a transcription, not a
   reinterpretation.** It must be proven to compute the same function as the
   rejected artifact (bit-identical outputs on a fixture set exercising every
   conditional branch the patch touches, an unchanged model revision and
   environment, a structural diff confined to the blocking construct). **The
   ORT reference is computed from the same artifact, so the GATE cannot detect
   a weakened re-export; only the comparison against the rejected artifact
   can.**
4. **The PROBE method.** Where the zero fill makes a declared output constant
   by construction (OWL-ViT's `input_ids == 0` mask fill), the probe records
   that fact and does not treat the degenerate output as evidence about the
   model; branch liveness is proved separately.

### Residuals for the next boundary

- **The ten missing-export groups** are the next tranche: EdgeCrafter
  detection / segmentation / pose (heaviest — upstream clone, new environment,
  checkpoint downloads), RF-DETR detection / segmentation / keypoint at other
  sizes (**not executable here** — `import rfdetr` fails under transformers
  4.35 and the weights auto-download, so these depend on maintainer decision
  1), ViTPose-base-simple, Depth-Anything-V2, VideoMAE/ViViT/TimeSformer
  (static 5D fixtures), RAFT-small (**exported at opset 16 by
  `raft_exporter.py`, so it is converted** — and that exporter also hardcodes
  `dynamic_axes`, so the packet must pin static shapes, not just the opset).
- **The CUDA 72 rows assume a cuDNN-enabled build** (`Conv`/`ConvTranspose`
  sit behind `#ifdef NERT_HAS_CUDNN`). This is now recorded in the matrix.
- **`nert_gate_cuda` was relinked once** (it predated `build-cudnn/libnert.a`).
  Every harness binary statically links `libnert.a` and must be relinked after
  any rebuild; the GPU leg must not run against a stale binary.
- **`onnxslim` is absent and cannot be installed** (no write access to the
  system site-packages), so Ultralytics exports are unsimplified. That is
  recorded in the provenance; it is not a defect.
- **Reviewer findings of the kind that recur**, now recorded for the next
  packets: an acceptance script that greps for a self-attested `EXIT=0`
  string rather than the run's own captured status; a script that counts
  delimiters without saying whether it counts escaped occurrences; and an
  evidence file whose `verdict` regex also accepts `BLOCKED`. All three were
  found this boundary; all three are avoidable.

### Tranche 2 — planned at this boundary, not executed

The tranche-2 decomposition was prepared at this boundary and is summarised
here because `/tmp` does not survive. **Seven of the ten groups are
executable on this host** (entries 19, 20, 21, 26, 27, 29, 31); the three
RF-DETR groups (22–24) are **not** — `import rfdetr` fails under transformers
4.35 and the weights auto-download from `storage.googleapis.com`, so they are
blocked on maintainer decision 1 and **nothing in the RF track should be
attempted without an explicit ruling**. The RF track is fully decoupled, so
decision 1 can never block the other seven groups.

**Recommended order** (at most two packets in flight, disjoint writable
paths, the GPU single-owner): (0) an **RF-DOC** packet that *proves* the
non-executability from verbatim transcripts — the three exporters' `--help`
showing the `--opset_version 17` default, the exact `import rfdetr` failure,
the installed transformers version, the non-hermetic weight evidence, and the
keypoint exporter's preview-fallback lines — plus **RAFT-PROV**; then
(1) ViTPose-PROV + Depth-Anything-PROV; (2) the three video exports
(one packet) alongside **EdgeCrafter-PROV** (the long pole: upstream clone,
a new environment, three checkpoint downloads); (3) the gates, **serialized
on the GPU**, shortest CPU leg first (RAFT-small, ViTPose, Depth), then the
three video models, then the three EdgeCrafter models last, so that any
re-provision cycle lands at the end of the queue; (4) **INT-2** alone.

**Method changes tranche 2 must carry** (all from tranche-1 findings):
- **Static 5D fixture discipline** for the video group — one fixture file,
  frozen in the provision packet's provenance, reused byte-identically for
  the ORT reference and all three legs, with **zero `dim_param`** on any
  gated artifact.
- **Image-pair fixture** for RAFT-small — a deterministic fixed shift of the
  first frame, both bound by declared name, with **static** H/W. The RAFT
  exporter hardcodes **both** opset 16 **and** `dynamic_axes`, so that
  packet pins shapes as well as the opset.
- **Multi-input binding including int64 payloads** — EdgeCrafter's
  `orig_target_sizes [1,2]` must be constructed and bound, not omitted.
- **External-data / self-containment check** asserted explicitly in every
  converted or externally-stored export's `verify_export.py`.
- **Acceptance hygiene** — assert per-leg exit statuses captured from the
  run, never a self-attested `EXIT=0` string; count **unescaped** pipes
  explicitly; make every evidence-verdict regex **exclude BLOCKED**; scope
  repo-cleanliness checks with `git -C`.
- **RAFT-small hedges decision 5** by recording the verbatim conversion
  caveat **and** an ORT-vs-ORT pre/post-conversion diff, so either ruling
  is satisfiable without re-running the gate.
- If all seven close, the scoreboard becomes **PASS 26 / Open 1 / Missing 3
  / Out of scope 2** (26+1+3 = 30).

### Scoreboard after this boundary

**PASS 19 / Open 1 (Grounding DINO, OpenCL backend gap) / Missing export 10 /
Out of scope 2** — 19 + 1 + 10 = 30 in scope. The release gate is **zero open
entries and zero missing exports** among the 30; **no release was cut**.
