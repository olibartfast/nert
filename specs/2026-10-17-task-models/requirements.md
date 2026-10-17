# Requirements — Other task models on NERT

Date: 2026-10-17. YOLO26n proved the pattern on CUDA (fully
dispatched, gate R-6 passed). This packet extends the same treatment
to every other task-model class Neuriplo needs, highest node count
first — the YOLO-ordered rule that already prioritizes the work.

## Where things stand

- Detection CNN (YOLO26n, 485 nodes): done on CPU and CUDA.
- Classification (ResNet-18): runs in CI on CPU; the CUDA run is
  pending but every op it needs already has a CUDA row.
- Transformer detection (DETR): waiting on an opset-18 export and the
  Neuriplo parity harness. Known NERT gap: LayerNorm is CPU-only, so
  the GPU side will need a new row.
- Open-vocab detection (OWLv2, Grounding-DINO): the Tier-5 CPU kernels
  (comparison, bitwise, reduction, and scatter families) already
  landed; parity still needs Neuriplo plus the model weights.
- Segmentation and pose models: no exports in hand; tracked as future
  work, not this packet.

## In scope

- For each model class: a CPU-vs-ORT parity run, a GPU dispatch-gap
  analysis (which nodes lack device rows), and closing those gaps in
  node-count order.
- A living coverage matrix (models × devices) so gaps stay visible.

## Out of scope

- Training, quantization, and new model architectures.
- Segmentation/pose models (no exports available).
- AMD and Vulkan backends for these models (blocked as before: AMD
  needs its runner, Vulkan needs a shader toolchain).

## Open questions

- Q-1: Where does each new GPU kernel live exactly once? The
  single-source decision (canonical HIP vs generated HIP vs policed
  mirrors) is still with the maintainer, and it gates every new GPU
  row below. CPU rows are unaffected.
- Q-2: Who supplies the opset-18 DETR and OWLv2 exports — Neuriplo
  side or NERT side?
- Q-3: ORT stays pinned at 1.27.0 for parity (the YOLO precedent), or
  move with Neuriplo's pin?
