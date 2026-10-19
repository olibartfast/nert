# Handoff — NERT catalog coverage, boundary 1 of N

Date: 2026-10-19. Written by the closing orchestrator session
`ses_eaca1c986ffeFJdfdPOFfK3mDh`. Read this file, the ledger, the inventory and
the coverage matrix; nothing else is needed to continue.

## Where the program stands

- Repo: `/home/oli/repos/nert`, branch `main`, pushed commits `6d99da9`,
  `b905792` (maintainer-committed), `346cb29`. Working tree clean.
- OpenCL operator table: **68 rows** (was 37). CPU 72, CUDA 72 unchanged.
- In-scope catalog target: **30 ONNX entries** of 32 (image understanding and
  Gaussian Splatting are maintainer-approved OUT OF SCOPE with rationale).
- Current scoreboard: **13 PASS**, 0 OPS-only, 2 gate-pending, 1 failed-gate,
  14 missing-export. The catalog is NOT complete and no release is to be cut.

## Scoreboard detail

PASS on CPU+CUDA+OpenCL with reviewer PASS, full gate evidence and commit IDs:
ResNet-18, YOLO26n, OWLv2, RF-DETR seg-small, RAFT-large, YOLO11s, YOLOv8n-pose,
YOLO26s-pose, RF-DETR nano-1101, YOLO26n-depth, RF-DETR keypoint-preview,
YOLO26s*, YOLO26x* (* R-6 tie-flip deviation, documented in coverage.md).

Open, with owners:
1. **Grounding DINO** — CPU full run completed (exit 0; outputs
   `/tmp/opencode/gdino/nert_cpu_logits.bin`, `nert_cpu_pred_boxes.bin`). Missing:
   the CUDA run, the ORT comparison of both legs, and the OpenCL leg. All 26 op
   types it uses are in the 68-row table, so no kernel work is expected.
2. **RT-DETRv4-s** — CPU and OpenCL PASS; CUDA GAP: node
   `/model/backbone/stem/stem2b/conv/Conv` has asymmetric pads `[0,0,1,1]` and is
   rejected by `src/cuda/conv_frontend.cu:320` and `src/cuda/conv.cu:317`
   (cuDNN needs symmetric padding). Minimal remedy: an explicit pre-pad on the
   CUDA Conv path.
3. **RF-DETR seg-nano-576** — CPU PASS, OpenCL and CUDA FAIL on ONE TopK
   near-tie at rank 99 (ORT picks 182, GPUs pick 1503; scores tied to 1.43e-06
   vs 4.82e-05 fp32 reduction noise). With the TopK selection pinned to ORT's,
   all three devices pass. The NERT selection rule is already identical to the
   oracle; the fix belongs in the export/topology, not in a kernel.
4. **14 missing ONNX exports** — export-first: provision opset-18, probe the op
   set, then add only the rows that graph needs. Highest leverage next:
   YOLOv8n detector, YOLO11m-seg (checkpoint local at
   `/home/oli/repos/tritonic/yolo11m-seg.pt`), OWL-ViT, ResNet-50, ViTPose,
   Depth-Anything-V2, VideoMAE/ViViT/TimeSformer, RAFT-small, YOLO v5-v12.
   Network is reachable (huggingface.co, github.com, pypi.org all 200);
   ultralytics 8.4.113, onnx 1.22, onnxruntime 1.27, transformers 4.35, torch 2.12.
5. **HIP** — unsupported for 0.2.0, evidenced (`which hipcc rocminfo amdclang++
   rocm-smi` empty, no `/opt/rocm`, only OpenCL platform is NVIDIA CUDA on an
   RTX 3060, HIP table still 9 rows at `src/hip/table.cpp:28-29`). No paper mirror.

## Reusable assets (do not rebuild)

- Gate harness: `/tmp/opencode/nert_gate.cpp` + `nert_gate_cpu`, `nert_gate_opencl`,
  `nert_gate_cuda` (CUDA needs `LD_LIBRARY_PATH=$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib:/tmp/opencode/cudnn/lib64`)
  + `/tmp/opencode/gate_compare.py` + `/tmp/opencode/build_gate_*.sh`.
- ORT references: `/tmp/opencode/catalog_refs/<model>/{input.npy, ref_<out>.npy}`
  (bus.jpg preprocessed, deterministic) for yolo26s, yolo26x, yolo11s,
  yolov8n_pose, yolo26s_pose, yolo26n_depth, rfdetr_nano_1101,
  rfdetr_seg_nano_576, rfdetr_keypoint_preview, rtv4_hgnetv2_s.
- Gate evidence: `/tmp/opencode/gates/*_evidence.md`.
- Row sources: `src/opencl/{math_unary_extra,reduce_extra,index_move,resize,grid_sample,norm_range_pad,math_extra2,index_extra,conv_transpose}.cpp`.

## Process rules for the next session

1. **Planner first**: dispatch `planner` before each model/family to produce the
   packet decomposition (writable paths, read-only interfaces, required final
   state, exact identifiers, permitted commands, ONE acceptance command, explicit
   stop condition). The orchestrator judges and dispatches; it does not invent
   packets.
2. **Reviewer mandatory**: after every implementer packet and before accepting
   it, dispatch a fresh `reviewer` (read-only, zero write access) to inspect the
   real diff, the writable-path list, the focused tests and the acceptance
   evidence. Record the reviewer session ID and verdict in the ledger. A packet
   is accepted only with reviewer PASS **and** a green acceptance command.
   Defects go back to a fresh implementer; never patch in place.
3. **Concurrency**: hard cap of 2 running packets; shared files
   (`src/opencl/table.cpp`, `CMakeLists.txt`, `test/CMakeLists.txt`,
   `coverage.md`, `CHANGELOG.md`) are owned by one serialized integration packet.
4. **Acceptance runs once** per attempt, as the final action; a failed attempt is
   reported, not looped. Every packet states what it must NOT do and when to stop.
5. **Context reset** at each model boundary: handoff file, stop the worker, new
   orchestrator session, new session ID recorded in the ledger.
6. Gate runners (`general`) are verification; they never substitute for a reviewer.
7. Commit scoped paths only; push to `main`; report commit IDs. No tags, no
   release, until the scoreboard has zero OPS/GAP/BLOCKED/MISSING-EXPORT among
   the 30 in-scope entries.

## Immediate next packets (planner to confirm/refine)

- **A** Grounding DINO gate completion: CUDA run, ORT comparison of the CPU and
  CUDA legs, then the OpenCL leg. Read-only on the repo; artifacts under
  `/tmp/opencode/gdino/`.
- **B** RT-DETRv4 CUDA asymmetric-pad Conv remedy (writable:
  `src/cuda/conv_frontend.cu`, `src/cuda/conv.cu` only), then re-gate.
- **C** RF-DETR seg-nano-576 disposition: decide export/topology remedy
  (re-export at higher precision, or pin the TopK selection) and re-gate.
- **D** Export-first batch: YOLOv8n detector, YOLO11m-seg, OWL-ViT, ResNet-50
  (provision + probe + gate), then only the rows the probed graphs need.
- **E** Serialized integration: coverage.md + CHANGELOG.md update, commit, push.
