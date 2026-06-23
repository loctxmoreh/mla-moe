#!/usr/bin/env python3
"""Reference hex dumper — the Python side of the weight-loading cross-check.

Reads raw bytes straight out of the safetensors shard (numpy has no bfloat16
dtype, and we want raw bits anyway) and writes the first N u16 words, one
lowercase %04x per line — byte-identical to `mla-moe dump`, so the two can be
diffed directly.

Usage:
    python dump_hex.py <shard_dir> <tensor_name> [n] [outfile]
    python dump_hex.py --list <shard_dir>      # tensors present in this checkpoint

Exit codes: 0 ok, 3 tensor absent from the checkpoint.
"""
import sys, json, struct, pathlib
import numpy as np

# Canonical tensor set for the cross-check (single source of truth; the driver
# discovers the subset present in a checkpoint via --list). Naming differs by
# model: dsv2lite has q_proj (no q-LoRA) and a dense layer 0; glm47 has
# q_a/q_b/q_a_layernorm. Absent names are simply skipped.
TENSORS_TO_CHECK = [
    "model.layers.0.self_attn.q_proj.weight",              # dsv2lite (no q-LoRA)
    "model.layers.0.self_attn.q_a_proj.weight",            # glm47 (q-LoRA)
    "model.layers.0.self_attn.q_b_proj.weight",            # glm47 (q-LoRA)
    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
    "model.layers.0.self_attn.kv_b_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.q_a_layernorm.weight",       # glm47 only
    "model.layers.0.self_attn.kv_a_layernorm.weight",
    "model.layers.1.mlp.experts.0.gate_proj.weight",       # first MoE layer
    "model.layers.1.mlp.gate.weight",
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
]


def load_weight_map(shard_dir):
    index = json.loads((shard_dir / "model.safetensors.index.json").read_text())
    return index["weight_map"]


def dump(shard_dir, tensor_name, n_req, outfile):
    weight_map = load_weight_map(shard_dir)
    shard = weight_map.get(tensor_name)
    if shard is None:
        sys.stderr.write(f"absent: {tensor_name}\n")
        return 3

    with open(shard_dir / shard, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        data_start = 8 + hlen
        entry = header[tensor_name]
        n = min(n_req, int(np.prod(entry["shape"])))
        begin, _end = entry["data_offsets"]
        f.seek(data_start + begin)
        raw = f.read(n * 2)

    u16 = np.frombuffer(raw, dtype="<u2")
    text = "".join(f"{int(v):04x}\n" for v in u16.tolist())
    if outfile and outfile != "-":
        pathlib.Path(outfile).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--list":
        weight_map = load_weight_map(pathlib.Path(sys.argv[2]))
        for name in TENSORS_TO_CHECK:
            if name in weight_map:
                print(name)
        return 0

    if len(sys.argv) < 3:
        sys.stderr.write(__doc__)
        return 1

    shard_dir   = pathlib.Path(sys.argv[1])
    tensor_name = sys.argv[2]
    n_req       = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    outfile     = sys.argv[4] if len(sys.argv) > 4 else None
    return dump(shard_dir, tensor_name, n_req, outfile)


if __name__ == "__main__":
    sys.exit(main())
