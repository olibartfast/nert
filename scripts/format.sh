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
    "include/engine/Device.hpp"
    "include/engine/Engine.hpp"
    "include/engine/Graph.hpp"
    "include/engine/ModelLoader.hpp"
    "include/engine/Shapes.hpp"
    "src/CpuDevice.cpp"
    "src/Engine.cpp"
    "src/kernels/Add.cpp"
    "src/kernels/Concat.cpp"
    "src/kernels/Div.cpp"
    "src/kernels/Expand.cpp"
    "src/kernels/Gather.cpp"
    "src/kernels/GatherElements.cpp"
    "src/kernels/Kernels.hpp"
    "src/kernels/Mul.cpp"
    "src/kernels/Split.cpp"
    "src/kernels/Sub.cpp"
    "src/kernels/Transpose.cpp"
    "src/kernels/Where.cpp"
    "src/ModelLoader.cpp"
    "src/ShapeInference.cpp"
    "src/WireReader.cpp"
    "src/WireReader.hpp"
    "test/DeviceTest.cpp"
    "test/KernelsTest.cpp"
    "test/LoaderTest.cpp"
    "test/MemoryPlannerTest.cpp"
    "test/ShapeInferenceTest.cpp"
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
    \( -name '*.cpp' -o -name '*.hpp' \) -print | sort)

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
