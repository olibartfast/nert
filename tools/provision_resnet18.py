#!/usr/bin/env python3
"""Provision the ResNet-18 test fixture for NERT.

Exports torchvision ResNet-18 to ``<output_dir>/resnet18.onnx`` with the
opset-18 operator set and verifies the emitted graph against the operators
NERT implements for this model. Needs torch and torchvision 0.27.0 (plus onnx
and onnxscript for the exporter).

Usage:
    provision_resnet18.py <output_dir>
"""

import os
import sys

# The exporter and the emitted graph are pinned so the fixture is reproducible.
PINNED_TORCHVISION_VERSION = "0.27.0"
OPSET_VERSION = 18

# Operators NERT implements for this model. The fixture must use only these:
# the opset-18 export expresses the ResNet-18 graph with ReduceMean/Reshape instead of the legacy
# GlobalAveragePool/Flatten/Identity set.
SUPPORTED_OPS = {
    "Add",
    "Conv",
    "Gemm",
    "MatMul",
    "MaxPool",
    "ReduceMean",
    "Relu",
    "Reshape",
}

# These legacy ops must not appear in the exported graph.
FORBIDDEN_OPS = {"Identity", "Flatten", "GlobalAveragePool"}


def fail(message):
    print(f"provision_resnet18: {message}", file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        fail(f"expected exactly one argument (output dir), got {len(sys.argv) - 1}")

    output_dir = os.path.abspath(sys.argv[1])
    if output_dir == "/workspace" or output_dir.startswith("/workspace/"):
        fail("refusing to write under /workspace")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "resnet18.onnx")

    try:
        import torch
        import torch.onnx
        import torchvision
        import torchvision.models as models
        import onnx
    except ImportError as exc:
        fail(f"required Python package missing: {exc}")

    version = getattr(torchvision, "__version__", "")
    base_version = version.split("+", 1)[0]
    if base_version != PINNED_TORCHVISION_VERSION:
        fail(
            "torchvision version mismatch: fixture export is pinned to "
            f"{PINNED_TORCHVISION_VERSION}, found {version!r} (base {base_version!r})"
        )

    model = models.resnet18(pretrained=True)
    model.eval()

    example_input = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            model_path,
            opset_version=OPSET_VERSION,
            input_names=["input"],
            output_names=["output"],
        )

    graph = onnx.load(model_path)

    opsets = {entry.domain or "": entry.version for entry in graph.opset_import}
    for domain, version in opsets.items():
        if version != OPSET_VERSION:
            fail(f"opset_import {domain!r} is version {version}, expected {OPSET_VERSION}")

    op_types = [node.op_type for node in graph.graph.node]
    op_set = set(op_types)

    forbidden = sorted(op_set & FORBIDDEN_OPS)
    if forbidden:
        fail(f"graph contains forbidden op(s): {forbidden}")

    unsupported = sorted(op_set - SUPPORTED_OPS)
    if unsupported:
        fail(f"graph contains op(s) the NERT does not implement: {unsupported}")

    histogram = {op: op_types.count(op) for op in sorted(op_set)}
    print(f"provisioned {os.path.abspath(model_path)}")
    print(f"nodes={len(op_types)} ops={histogram}")


if __name__ == "__main__":
    main()
