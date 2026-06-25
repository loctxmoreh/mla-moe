"""Compare the C engine's unabsorbed-prefill dumps against the oracle.

    uv run python compare_prefill.py <c_dump_dir> [dsv2lite|glm47]

C dumps are raw little-endian named "<tensor>.bin"; oracle tensors are loaded by
name via load_oracle (manifest-driven). Reports max-abs-err per tensor against
the model's tol (manifest "tol"). Non-finite entries (causal-mask -inf in
attn_scores) are compared only where both sides are finite.
"""
import os
import sys

import numpy as np

import load_oracle

C_DIR = sys.argv[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "dsv2lite"
ORACLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps", MODEL)

man = load_oracle.manifest(ORACLE_DIR)
TOL = man.get("tol", 2e-3)
n_layers = man["config"]["n_layers"]

names = ["prefill_hidden_states"]
for i in range(n_layers):
    names += [f"prefill_layer{i:02d}_attn_out", f"prefill_layer{i:02d}_mlp_out"]
names += [f"prefill_layer0_{x}" for x in
          ("q_nope", "q_pe", "c_kv", "k_pe", "k_nope", "value",
           "attn_scores", "attn_weights")]
names += ["prefill_moe1_router_scores", "prefill_moe1_topk_idx",
          "prefill_moe1_topk_w", "prefill_logits"]


def load_c(name, shape, is_int):
    path = os.path.join(C_DIR, name + ".bin")
    if not os.path.exists(path):
        return None
    dt = np.dtype("<i4" if is_int else "<f4")
    return np.fromfile(path, dtype=dt).reshape(shape)


worst = 0.0
n_fail = n_miss = 0
for name in names:
    if name not in man["tensors"]:
        continue
    is_int = man["tensors"][name]["dtype"] == "int32"
    oracle = load_oracle.load(name, ORACLE_DIR)
    c = load_c(name, oracle.shape, is_int)
    if c is None:
        print(f"  MISS  {name}")
        n_miss += 1
        continue
    if name == "prefill_moe1_topk_idx":
        # order-insensitive set match per row (tie-break may differ)
        ok = all(set(a) == set(b) for a, b in zip(oracle, c))
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:34s} {'sets match' if ok else 'sets DIFFER'}")
        n_fail += not ok
        continue
    o = oracle.astype(np.float64)
    cc = c.astype(np.float64)
    mask = np.isfinite(o) & np.isfinite(cc)
    err = np.abs(o[mask] - cc[mask]).max() if mask.any() else 0.0
    worst = max(worst, err)
    status = "ok  " if err <= TOL else "FAIL"
    n_fail += err > TOL
    print(f"  {status}  {name:34s} max_abs_err={err:.3e}")

# bonus: does argmax of last-position logits match the oracle's next_id?
log = load_c("prefill_logits", load_oracle.load("prefill_logits", ORACLE_DIR).shape, False)
if log is not None:
    pred = int(log[-1].argmax())
    nid = man["next_id"]
    print(f"\n  argmax(last logits) = {pred}  (oracle next_id = {nid})  "
          f"{'MATCH' if pred == nid else 'MISMATCH'}")

print(f"\n  tol={TOL:.1e}  worst_err={worst:.3e}  failures={n_fail}  missing={n_miss}")
sys.exit(1 if (n_fail or n_miss) else 0)
