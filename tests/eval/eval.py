"""Score the C engine against the frozen golden dataset.

    uv run python eval.py [dsv2lite|glm47] [options]

Three metrics, climbing the precision ladder, all scored against the frozen
`<model>/` dataset produced by gen_reference.py:

  1. teacher-forced top-1 agreement  (headline "accuracy") -- needs no HF
  2. perplexity relative error       (C ppl vs frozen HF nll) -- needs no HF
  3. free-run fuzzy (METEOR/BERTScore, optional --fuzzy) -- lexical/semantic

Top-1 is scored over the COMPLETION region only (pos >= prompt_len): prompt
positions measure natural-language unpredictability, not engine correctness.
Both engine paths are scored: 'P' (prefill/unabsorbed) and 'D' (decode/absorbed).
A mismatch with logit gap <= --tie is a numerical tie, not an error (tie-tolerant
column); the strict column gates. Thresholds come from threshold.json.

Options:
  -r, --run PATH   C run binary (default <repo>/run)
  --tie FLOAT      logit-gap tie threshold (default 2e-3, the oracle budget)
  --thresholds P   threshold.json (default <here>/threshold.json)
  --fuzzy          also run the free-run METEOR/BERTScore tier (heavy deps)
  --max-new INT    tokens to free-run in the fuzzy tier (default = dataset max_new)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default="dsv2lite", choices=["dsv2lite", "glm47"])
    p.add_argument("-d", "--dir", default=None, help="dataset dir (default <here>/<model>)")
    p.add_argument("-r", "--run", default=os.path.join(_REPO, "run"))
    p.add_argument("--model-dir", default=None,
                   help="override the model dir (reference.json's is provenance; "
                        "its absolute path won't exist on another machine)")
    p.add_argument("--tie", type=float, default=2e-3)
    p.add_argument("--thresholds", default=os.path.join(_HERE, "threshold.json"))
    p.add_argument("--fuzzy", action="store_true")
    p.add_argument("--max-new", type=int, default=None)
    return p.parse_args()


def read_id_lines(path):
    with open(path) as f:
        return [[int(x) for x in ln.split()] for ln in f if ln.strip()]


def write_i32(ids):
    import struct
    f = tempfile.NamedTemporaryFile(suffix=".i32.bin", delete=False)
    f.write(struct.pack("<%di" % len(ids), *ids))
    f.close()
    return f.name


def run_c(run_bin, model_dir, ids, *mode):
    path = write_i32(ids)
    try:
        return subprocess.run([run_bin, model_dir, path, *map(str, mode)],
                              capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(path)


def parse_teacher(out, prompt_len):
    """-> {'P': (strict, tie_tol, n, misses), 'D': (...)} over completion region."""
    rows = {"P": [], "D": []}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] in ("P", "D"):
            tag, pos, gold, am, gap = parts
            rows[tag].append((int(pos), int(gold), int(am), float(gap)))
    return rows


def score_rows(rows, prompt_len, tie):
    comp = [(pos, gold, am, gap) for (pos, gold, am, gap) in rows if pos >= prompt_len]
    if not comp:
        return None
    strict = sum(am == gold for _, gold, am, _ in comp)
    tol = sum(am == gold or gap <= tie for _, gold, am, gap in comp)
    misses = [(pos, gold, am, gap) for (pos, gold, am, gap) in comp if am != gold]
    return strict / len(comp), tol / len(comp), len(comp), misses


def parse_ppl(out):
    c_nll = c_ntok = None
    for line in out.splitlines():
        if line.startswith("nll "):
            _, nll, _, ntok = line.split()
            c_nll, c_ntok = float(nll), int(ntok)
    return c_nll, c_ntok


def main():
    import math
    args = parse_args()
    if not os.path.exists(args.run):
        sys.exit(f"C run binary not found: {args.run} (build with `make run`)")
    data = args.dir or os.path.join(_HERE, args.model)
    ref = json.load(open(os.path.join(data, "reference.json")))
    model_dir = args.model_dir or ref["model_dir"]
    thr = json.load(open(args.thresholds))
    prompts = read_id_lines(os.path.join(data, "prompts.i32.txt"))
    comps = read_id_lines(os.path.join(data, "completions.i32.txt"))
    recs = ref["requests"]
    n = len(recs)
    assert len(prompts) == len(comps) == n, "dataset length mismatch"

    print(f"[{args.model}] {model_dir}  ({n} requests)  tie={args.tie:.1e}", flush=True)
    print(f"  gates: top1_strict>={thr['top1_strict']}  ppl_rel<={thr['ppl_rel']}"
          + (f"  meteor>={thr['meteor']}  bertscore_f1>={thr['bertscore_f1']}" if args.fuzzy else ""),
          flush=True)

    tot = {"P_ok": 0, "D_ok": 0, "cmp": 0}          # top-1 aggregates
    worst_ppl = 0.0
    all_misses = []
    for i, (pids, cids, rec) in enumerate(zip(prompts, comps, recs)):
        full = pids + cids
        plen = len(pids)
        # --- Tier 1+2: teacher-forced top-1, both paths ---
        out = run_c(args.run, model_dir, full, "teacher", plen)
        rows = parse_teacher(out, plen)
        sp = score_rows(rows["P"], plen, args.tie)
        sd = score_rows(rows["D"], plen, args.tie)
        # --- Tier 2: perplexity rel-err vs frozen HF nll ---
        c_nll, c_ntok = parse_ppl(run_c(args.run, model_dir, full, "ppl"))
        hf_ppl = math.exp(rec["hf_nll"] / rec["hf_ntok"])
        c_ppl = math.exp(c_nll / c_ntok)
        rel = abs(c_ppl - hf_ppl) / hf_ppl
        worst_ppl = max(worst_ppl, rel)
        if sd:
            tot["D_ok"] += round(sd[0] * sd[2]); tot["cmp"] += sd[2]
            all_misses += [(i, *m) for m in sd[3]]
        if sp:
            tot["P_ok"] += round(sp[0] * sp[2])
        print(f"  [{i}] comp={sd[2] if sd else 0:4d}  "
              f"P_top1={sp[0]*100:6.2f}% (tie {sp[1]*100:6.2f}%)  "
              f"D_top1={sd[0]*100:6.2f}% (tie {sd[1]*100:6.2f}%)  "
              f"ppl C={c_ppl:.4f} HF={hf_ppl:.4f} rel={rel:.2e}", flush=True)

    cmp = tot["cmp"] or 1
    d_strict = tot["D_ok"] / cmp
    p_strict = tot["P_ok"] / cmp
    print()
    print(f"  decode-path  top1_strict = {d_strict*100:.3f}%  ({tot['D_ok']}/{cmp})")
    print(f"  prefill-path top1_strict = {p_strict*100:.3f}%")
    print(f"  worst ppl rel-err        = {worst_ppl:.3e}")
    if all_misses:
        print(f"  decode misses (pos, gold, argmax, gap), worst first:")
        for req, pos, gold, am, gap in sorted(all_misses, key=lambda m: -m[4])[:10]:
            print(f"    req{req} pos{pos}: gold {gold} != argmax {am}  gap={gap:.4f}"
                  + ("  [tie]" if gap <= args.tie else ""))

    ok = d_strict >= thr["top1_strict"] and worst_ppl <= thr["ppl_rel"]

    if args.fuzzy:
        ok = run_fuzzy(args, model_dir, prompts, comps, recs, thr) and ok

    print()
    print("  RESULT:", "ok" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def run_fuzzy(args, model_dir, prompts, comps, recs, thr):
    """Free-run greedy generation vs golden completion, scored METEOR+BERTScore."""
    from transformers import AutoTokenizer
    import evaluate  # noqa
    tok = AutoTokenizer.from_pretrained(model_dir)
    preds, refs = [], []
    for pids, cids, rec in zip(prompts, comps, recs):
        max_new = args.max_new or rec["completion_len"]
        out = run_c(args.run, model_dir, pids, "gen", max_new)
        gen_ids = []
        for line in out.splitlines():
            if line.startswith("completion"):
                gen_ids = [int(x) for x in line.split()[1:]]
        preds.append(tok.decode(gen_ids))
        refs.append(tok.decode(cids))
    meteor = evaluate.load("meteor").compute(predictions=preds, references=refs)["meteor"]
    bs = evaluate.load("bertscore").compute(predictions=preds, references=refs, lang="en")
    f1 = sum(bs["f1"]) / len(bs["f1"])
    print(f"  METEOR = {meteor:.4f}  BERTScore-F1 = {f1:.4f}")
    return meteor >= thr["meteor"] and f1 >= thr["bertscore_f1"]


if __name__ == "__main__":
    main()
