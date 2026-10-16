#!/usr/bin/env bash
# Format check for NERT C++ sources.
# Usage:
#   ./scripts/format.sh          # check (--dry-run --Werror)
#   ./scripts/format.sh --fix    # rewrite files in place
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=check

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) MODE=check; shift ;;
        --fix) MODE=fix; shift ;;
        -h | --help)
            echo "Usage: $0 [--check|--fix]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if command -v clang-format-18 >/dev/null 2>&1; then
    CFMT=clang-format-18
elif command -v clang-format >/dev/null 2>&1; then
    CFMT=clang-format
else
    echo "clang-format not found. Install clang-format-18 (Ubuntu: apt install clang-format-18)." >&2
    exit 1
fi

# Files carried over from the Neuriplo tree that were never covered by its
# format check (Neuriplo's scope did not include engine/). They are not yet
# clang-format clean and stay outside the check until they are reformatted;
# remove an entry once its file is formatted.
EXCLUDE=(
    "include/nert/device.hpp"
    "include/nert/engine.hpp"
    "include/nert/graph.hpp"
    "include/nert/model_loader.hpp"
    "include/nert/shapes.hpp"
    "src/cpu_device.cpp"
    "src/engine.cpp"
    "src/kernels/add.cpp"
    "src/kernels/concat.cpp"
    "src/kernels/div.cpp"
    "src/kernels/expand.cpp"
    "src/kernels/gather.cpp"
    "src/kernels/gather_elements.cpp"
    "src/kernels/kernels.hpp"
    "src/kernels/mul.cpp"
    "src/kernels/split.cpp"
    "src/kernels/sub.cpp"
    "src/kernels/transpose.cpp"
    "src/kernels/where.cpp"
    "src/model_loader.cpp"
    "src/shape_inference.cpp"
    "src/wire_reader.cpp"
    "src/wire_reader.hpp"
    "test/device_test.cpp"
    "test/kernels_test.cpp"
    "test/loader_test.cpp"
    "test/memory_planner_test.cpp"
    "test/shape_inference_test.cpp"
)

is_excluded() {
    local f
    for f in "${EXCLUDE[@]}"; do
        [[ "$1" == "$ROOT/$f" ]] && return 0
    done
    return 1
}

FILES=()
while IFS= read -r f; do
    is_excluded "$f" || FILES+=("$f")
done < <(find "$ROOT/src" "$ROOT/include" "$ROOT/test" "$ROOT/bench" \
    \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' \) -print | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No C++ sources found." >&2
    exit 1
fi

if [[ "$MODE" == fix ]]; then
    echo "[$CFMT] Formatting ${#FILES[@]} file(s)..."
    "$CFMT" -i "${FILES[@]}"
    echo "Format complete."
else
    echo "[$CFMT] Checking ${#FILES[@]} file(s) (--dry-run --Werror)..."
    "$CFMT" --dry-run --Werror "${FILES[@]}"
    echo "Format check passed."
fi
