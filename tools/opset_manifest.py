"""Reviewed implementation manifest for the ONNX schema catalog.

This is development-only data used by ``tools/gen_opset_compat.py``. It is
deliberately explicit: every CPU-dispatched operator names the attributes the
loader, shape inference, and kernels actually implement. The generator derives
all other schema attributes as load-time guards, so a historical or newer
schema cannot silently acquire an unimplemented attribute.

The form/attribute review tables are audit checkpoints. If a new ONNX release
changes an input/output signature, an attribute type/default, or a value
domain, generation fails until the change is reviewed here.
"""

from __future__ import annotations

# Attribute names implemented by the runtime. Empty tuples are operators that
# take no attributes.
IMPLEMENTED_ATTRIBUTES = {
    "Abs": (),
    "Add": (),
    "And": (),
    "ArgMax": ("axis", "keepdims", "select_last_index"),
    "AveragePool": ("kernel_shape", "pads", "strides", "ceil_mode", "count_include_pad", "auto_pad"),
    "Cast": ("to",),
    "Clip": (),
    "Concat": ("axis",),
    "ConstantOfShape": ("value",),
    "Conv": ("pads", "strides", "dilations", "group", "auto_pad", "kernel_shape"),
    "ConvTranspose": ("pads", "strides", "dilations", "group", "kernel_shape", "output_padding", "auto_pad"),
    "Cos": (),
    "CumSum": ("exclusive", "reverse"),
    "Div": (),
    "Einsum": ("equation",),
    "Elu": ("alpha",),
    "Equal": (),
    "Erf": (),
    "Exp": (),
    "Expand": (),
    "EyeLike": ("dtype", "k"),
    "Flatten": ("axis",),
    "Floor": (),
    "Gather": ("axis",),
    "GatherElements": ("axis",),
    "Gemm": ("alpha", "beta", "transA", "transB"),
    "GlobalAveragePool": (),
    "Greater": (),
    "GreaterOrEqual": (),
    "GridSample": ("mode", "padding_mode", "align_corners"),
    "Identity": (),
    "InstanceNormalization": ("epsilon",),
    "IsNaN": (),
    "LayerNormalization": ("axis", "epsilon", "stash_type"),
    "Less": (),
    "LessOrEqual": (),
    "Log": (),
    "MatMul": (),
    "Max": (),
    "MaxPool": ("kernel_shape", "pads", "strides", "dilations", "ceil_mode", "storage_order", "auto_pad"),
    "Min": (),
    "Mod": ("fmod",),
    "Mul": (),
    "Neg": (),
    "Sign": (),
    "Not": (),
    "Or": (),
    "Pad": ("mode",),
    "Pow": (),
    "Range": (),
    "ReduceL2": ("keepdims", "noop_with_empty_axes", "axes"),
    "ReduceMax": ("keepdims", "axes", "noop_with_empty_axes"),
    "ReduceMean": ("keepdims", "noop_with_empty_axes", "axes"),
    "ReduceSum": ("keepdims", "noop_with_empty_axes", "axes"),
    "Relu": (),
    "Reshape": ("allowzero",),
    # `axes` is intentionally absent: the schema-18 Resize attribute is not
    # consumed by shape inference or the resize kernel, so it is guarded.
    "Resize": (
        "coordinate_transformation_mode",
        "cubic_coeff_a",
        "mode",
        "nearest_mode",
        "antialias",
        "exclude_outside",
        "extrapolation_value",
        "keep_aspect_ratio_policy",
    ),
    "ScatterND": ("reduction",),
    "Shape": ("start", "end"),
    "Sigmoid": (),
    "Sin": (),
    "Slice": (),
    "Softmax": ("axis",),
    "Split": ("axis", "split", "num_outputs"),
    "Sqrt": (),
    "Squeeze": ("axes",),
    "Sub": (),
    "Tanh": (),
    "Tile": (),
    "TopK": ("axis", "largest", "sorted"),
    "Transpose": ("perm",),
    "Unsqueeze": ("axes",),
    "Where": (),
}

# Whole-schema rejections: (first_version, last_version_or_None, reason).
BEHAVIOR_CHANGES = {
    "GridSample": (
        (20, None, "GridSample mode default and value domain changed at opset 20"),
    ),
    "Resize": (
        (10, 10, "the pre-11 two-input scales form is not implemented"),
    ),
    "Tile": (
        (1, 1, "the pre-6 three-input tile form is not implemented"),
    ),
}

# Value-domain guards on an otherwise implemented attribute:
# op -> ((first_version, last_version_or_None, ((attribute, value), ...), reason), ...)
VALUE_DOMAIN_GUARDS = {
    "Pad": (
        (19, None, (("mode", "wrap"),), "Pad gained mode='wrap' at opset 19"),
    ),
    "ScatterND": (
        (
            16,
            None,
            (("reduction", "add"), ("reduction", "mul")),
            "ScatterND only implements reduction='none'",
        ),
    ),
}

# Every input/output signature change in the schema history must be reviewed.
FORM_REVIEWS = {
    ("Clip", 11): "attribute min/max moved to optional inputs; input form implemented",
    ("Gemm", 11): "C became optional; the two-input form is implemented",
    ("MaxPool", 8): "Indices output is optional; only the first output is implemented and output arity rejects Indices",
    ("Pad", 11): "attribute paddings/value moved to inputs; input form implemented",
    ("Pad", 18): "optional axes input implemented by shape inference and the pad kernel",
    ("ReduceL2", 18): "axes became an optional input; both forms are implemented",
    ("ReduceMax", 18): "axes became an optional input; both forms are implemented",
    ("ReduceMean", 18): "axes became an optional input; both forms are implemented",
    ("ReduceSum", 13): "axes became an optional input; both forms are implemented",
    ("Reshape", 5): "shape moved from an attribute to an input; the input form is implemented",
    ("Resize", 11): "roi/scales/sizes input layout implemented",
    ("Resize", 13): "roi and scales became optional; implemented",
    ("Slice", 10): "attribute starts/ends/axes moved to inputs; input form implemented",
    ("Split", 2): "removed the optional split input; the attribute form is implemented",
    ("Split", 13): "split became an optional input; both forms are implemented",
    ("Squeeze", 13): "axes became an optional input; both forms are implemented",
    ("Tile", 6): "removed the separate axis input; the repeats input form is implemented",
    ("TopK", 10): "k moved from an attribute to an input; implemented",
    ("Unsqueeze", 13): "axes became a required input; both forms are implemented",
}

# Attribute type/default/required changes are behavior-sensitive; review each.
ATTRIBUTE_CHANGE_REVIEWS = {
    ("Cast", 6): "`to` changed from string to int; pre-13 schemas are not selectable at declaration 11+",
    ("Clip", 6): "min/max acquired explicit float limits; the same clamp behavior applies to the guarded pre-11 form",
    ("Concat", 4): "axis became required; valid models carry it",
    ("GridSample", 20): "mode default/value domain changed; whole schema is rejected",
    ("Softmax", 13): "axis default changed from 1 to -1; the kernel resolves the schema",
    ("Split", 2): "axis acquired its explicit default 0",
}
