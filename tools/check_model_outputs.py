#!/usr/bin/env python3
"""Check raw model outputs against frozen ref_<name>.npy files.

Usage: python3 tools/check_model_outputs.py REF_DIR OUTPUT_PREFIX [--tol 1e-3]
Float outputs must be finite and satisfy tol*(1+abs(reference)); integer and
boolean outputs must match exactly. Missing outputs and empty reference sets fail.
Requires NumPy for development validation only.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("output_prefix")
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()
    if not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("--tol must be finite and positive")
    paths = sorted(args.reference_dir.glob("ref_*.npy"))
    if not paths:
        parser.error("no reference outputs found")
    passed = True
    results = {}
    for path in paths:
        name = path.stem[4:]
        try:
            ref = np.load(path, allow_pickle=False)
            if ref.dtype not in (np.dtype("float32"), np.dtype("int64"), np.dtype("bool")):
                raise ValueError(f"unsupported reference dtype {ref.dtype}")
            output = Path(args.output_prefix + "_" + name.replace("/", "_") + ".bin")
            if output.stat().st_size != ref.nbytes:
                raise ValueError("output byte count differs from reference")
            got = np.fromfile(output, dtype=ref.dtype).reshape(ref.shape)
            if ref.dtype.kind == "f":
                delta = np.abs(got.astype(np.float64) - ref.astype(np.float64))
                finite = np.isfinite(got) & np.isfinite(ref)
                mismatch = int(np.count_nonzero(~finite | (delta > args.tol * (1 + np.abs(ref.astype(np.float64))))))
                results[name] = {"mismatch": mismatch, "elements": ref.size,
                                 "max_abs": float(delta.max()) if delta.size and finite.all() else None}
            else:
                mismatch = int(np.count_nonzero(got != ref))
                results[name] = {"mismatch": mismatch, "elements": ref.size}
            passed &= mismatch == 0
        except (OSError, ValueError) as error:
            results[name] = {"error": str(error)}
            passed = False
    print(json.dumps({"pass": bool(passed), "outputs": results}, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
