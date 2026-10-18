# Task-Model Coverage on NERT

Date: 2026-10-17.

## Purpose

YOLO26n established the pattern for running a real task model on NERT,
recorded as gate `R-6`: load the exported graph, match ONNX Runtime on CPU,
identify the operators that lack a device row, and close those gaps in
node-count order. This specification extends that pattern to the other model
families Neuriplo needs.

The live, model-by-device result is maintained in
[`coverage.md`](coverage.md). That matrix, not this document, is the source of
truth for which models currently pass, which have operator coverage but no
full-model gate, and which are blocked on an export.

## Scope

For each model family, this work covers:

- a CPU-versus-ONNX-Runtime parity run on a concrete export;
- a dispatch-gap analysis listing the operators that have no row on the
  target device;
- implementation of the missing device rows, ordered by the number of graph
  nodes they unlock;
- a full-model gate on each device that can run the graph.

The work also maintains the model-by-device coverage matrix so that a missing
check becomes an explicit roadmap item rather than an invisible gap.

## Out of Scope

- Training, quantization, and new model architectures.
- Segmentation and pose models for which no export exists. They remain future
  work until an export is available.
- AMD and Vulkan device layers for these models. AMD needs its runner;
  Vulkan needs a shader toolchain.

## Open Questions

- **`Q-1` — Where should each GPU kernel have its single source?** The choice
  between a canonical HIP implementation, generated HIP, and policed mirrors
  is still with the maintainer. It affects every new GPU row; CPU work can
  proceed independently.
- **`Q-2` — Who supplies the opset-18 DETR and OWLv2 exports?** The export can
  come from the Neuriplo side or from NERT.
- **`Q-3` — Which ONNX Runtime version defines parity?** The YOLO work used
  1.27.0. The alternative is to follow Neuriplo's current pin.
