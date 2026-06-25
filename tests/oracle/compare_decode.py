"""Compare the C engine's absorbed-decode dumps against the oracle.

    uv run python compare_decode.py <c_dump_dir> [dsv2lite|glm47]

The C decode must reproduce the oracle's single absorbed step (which gen_oracle
asserts equals the unabsorbed reference and HF). Matching it therefore confirms
the absorbed algebra transitively. Layer-0 dumps are heads-major like the oracle.
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

names = ["decode_layer0_c_kv_cache", "decode_layer0_k_pe_cache",
         "decode_layer0_q_absorbed", "decode_layer0_ctx_latent",
         "decode_layer0_attn_weights", "decode_layer0_attn_out", "decode_logits"]

worst = 0.0
n_fail = n_miss = 0
for name in names:
    if name not in man["tensors"]:
        continue
    oracle = load_oracle.load(name, ORACLE_DIR)
    path = os.path.join(C_DIR, name + ".bin")
    if not os.path.exists(path):
        print(f"  MISS  {name}"); n_miss += 1; continue
    c = np.fromfile(path, dtype="<f4").reshape(oracle.shape)
    err = np.abs(oracle.astype(np.float64) - c.astype(np.float64)).max()
    worst = max(worst, err)
    status = "ok  " if err <= TOL else "FAIL"
    n_fail += err > TOL
    print(f"  {status}  {name:32s} max_abs_err={err:.3e}")

# does decode logits argmax match HF's decode-step prediction?
dl = np.fromfile(os.path.join(C_DIR, "decode_logits.bin"), dtype="<f4")
oracle_dl = load_oracle.load("decode_logits", ORACLE_DIR).ravel()
print(f"\n  C decode argmax = {int(dl.argmax())}   "
      f"oracle decode argmax = {int(oracle_dl.argmax())}   "
      f"{'MATCH' if dl.argmax() == oracle_dl.argmax() else 'MISMATCH'}")
print(f"\n  tol={TOL:.1e}  worst_err={worst:.3e}  failures={n_fail}  missing={n_miss}")
sys.exit(1 if (n_fail or n_miss) else 0)
