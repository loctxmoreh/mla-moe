#!/usr/bin/env bash
#
# Cross-check the C weight loader against the Python reference by dumping each
# tensor to hex from both and diffing. The tensor set comes from dump_hex.py
# (--list), filtered to those present in the given checkpoint.
#
# Usage:
#   tests/weight_loading/compare_dumps.sh <model_shortname> <shard_dir> [n_elems]
#
# Example:
#   tests/weight_loading/compare_dumps.sh dsv2lite /path/to/DeepSeek-V2-Lite
#
set -u

MODEL="${1:?usage: compare_dumps.sh <model_shortname> <shard_dir> [n_elems]}"
SHARD_DIR="${2:?usage: compare_dumps.sh <model_shortname> <shard_dir> [n_elems]}"
N="${3:-64}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$ROOT/mla-moe"
DUMP_PY="$ROOT/tests/weight_loading/dump_hex.py"
IDX="$SHARD_DIR/model.safetensors.index.json"
OUT="$ROOT/tests/weight_loading/dumps/$MODEL"

if command -v uv >/dev/null 2>&1 && [ -f "$ROOT/pyproject.toml" ]; then
    PY=(uv run --project "$ROOT" python)
elif [ -x "$ROOT/.venv/bin/python" ]; then
    PY=("$ROOT/.venv/bin/python")
else
    PY=(python3)
fi

if [ ! -x "$CLI" ]; then
    echo "error: $CLI not built. Run: make CC=gcc mla-moe" >&2
    exit 1
fi

mkdir -p "$OUT"

mapfile -t TENSORS < <("${PY[@]}" "$DUMP_PY" --list "$SHARD_DIR")
if [ "${#TENSORS[@]}" -eq 0 ]; then
    echo "error: no checked tensors present in $SHARD_DIR" >&2
    exit 1
fi

pass=0; fail=0
for T in "${TENSORS[@]}"; do
    safe="$(printf '%s' "$T" | tr -c 'a-zA-Z0-9_' '_')"
    cfile="$OUT/$safe.c.hex"
    pfile="$OUT/$safe.py.hex"

    if ! "${PY[@]}" "$DUMP_PY" "$SHARD_DIR" "$T" "$N" "$pfile"; then
        printf 'FAIL  %-55s (python dump error)\n' "$T"; fail=$((fail+1)); continue
    fi
    if ! "$CLI" dump "$IDX" "$SHARD_DIR" "$T" "$N" "$cfile" 2>/dev/null; then
        printf 'FAIL  %-55s (cli dump error)\n' "$T"; fail=$((fail+1)); continue
    fi
    if diff -q "$cfile" "$pfile" >/dev/null; then
        printf 'PASS  %-55s (%s words)\n' "$T" "$(wc -l < "$cfile" | tr -d ' ')"
        pass=$((pass+1))
    else
        printf 'FAIL  %-55s (hex mismatch)\n' "$T"
        diff "$cfile" "$pfile" | head -8 | sed 's/^/      /'
        fail=$((fail+1))
    fi
done

echo "------------------------------------------------------------"
echo "$MODEL: pass=$pass fail=$fail  (n=$N, dumps in $OUT)"
[ "$fail" -eq 0 ]
