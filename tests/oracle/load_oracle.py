"""Load oracle dumps by name. Used by C-vs-oracle comparison scripts.

    from load_oracle import load
    logits = load("decode_logits")                    # dsv2lite (default)
    logits = load("decode_logits", "dumps/glm47")     # other model

Raw layout is C-order little-endian (float32 or int32) — read it directly from C
with fread into a row-major buffer of the manifest shape.
"""

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DUMP_DIR = os.path.join(_HERE, "dumps", "dsv2lite")


def manifest(dump_dir=DUMP_DIR):
    with open(os.path.join(dump_dir, "manifest.json")) as f:
        return json.load(f)


def load(name, dump_dir=DUMP_DIR):
    m = manifest(dump_dir)["tensors"][name]
    dt = np.dtype("<f4" if m["dtype"] == "float32" else "<i4")
    return np.fromfile(os.path.join(dump_dir, m["file"]), dtype=dt).reshape(m["shape"])


if __name__ == "__main__":
    import sys
    d = os.path.join(_HERE, "dumps", sys.argv[1]) if len(sys.argv) > 1 else DUMP_DIR
    m = manifest(d)
    print(f"model={m['model']} prompt={m['prompt']!r} seq={m['seq_len']} "
          f"next_id={m['next_id']}  ({len(m['tensors'])} tensors)")
    for name in m["tensors"]:
        print(f"  {name:34s} {m['tensors'][name]['shape']} {m['tensors'][name]['dtype']}")
