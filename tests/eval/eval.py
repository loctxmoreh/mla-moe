"""Score an engine against the frozen golden dataset.

    uv run python eval.py [dsv2lite|glm47] [options]           # drive `run` directly
    uv run python eval.py [dsv2lite|glm47] --tokens out.txt    # score a token-ids file

Two entry points onto the same frozen `<model>/` dataset (built by gen_reference.py):

A. DEFAULT -- drives the C `run` binary through its single-sequence eval modes:

  1. teacher-forced top-1 agreement  (headline "accuracy") -- needs no HF
  2. perplexity relative error       (C ppl vs frozen HF nll) -- needs no HF
  3. free-run fuzzy (METEOR/BERTScore, optional --fuzzy) -- lexical/semantic

  Top-1 is scored over the COMPLETION region only (pos >= prompt_len): prompt
  positions measure natural-language unpredictability, not engine correctness.
  Both engine paths are scored: 'P' (prefill/unabsorbed) and 'D' (decode/absorbed).
  A mismatch with logit gap <= --tie is a numerical tie, not an error (tie-tolerant
  column); the strict column gates.

B. --tokens FILE -- scores a generated-token-ids file (one line of space-separated
  ids per request) against completions.i32.txt. This is the getp gate: the batch
  harness (src/getp_eval.c) writes exactly this file after timing inference(), so
  the graded engine is scored on ITS OWN output. It makes no assumption about how
  those ids were produced -- batched, continuous-batched, or one request at a time.

  The GATE is the announced accuracy gate: METEOR + BERTScore-F1. Free-run prefix
  agreement (tokens before the first divergence from the golden continuation) is
  printed alongside as a DIAGNOSTIC and does not affect the verdict -- an engine
  using bf16/fp8 weights or a bf16 KV cache legitimately diverges from the fp32
  reference, so sameness cannot gate. Read the prefix numbers to tell a
  numerically-different engine from a broken one, and see score_nll.py for the
  measurement that actually separates those two cases.

Thresholds for both come from threshold.json.

Options:
  -r, --run PATH   C run binary (default <repo>/run)
  --tokens PATH    score this generated-ids file instead of driving `run`
  --quick          --tokens: diagnostics only, skip the accuracy tier (exits 2)
  --tie FLOAT      logit-gap tie threshold (default 2e-3, the oracle budget)
  --thresholds P   threshold.json (default <here>/threshold.json)
  --fuzzy          path A only: also run the METEOR/BERTScore tier (heavy deps)
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
    p.add_argument("--tokens", default=None,
                   help="score this generated-ids file (getp output) instead of "
                        "driving the run binary")
    p.add_argument("--model-dir", default=None,
                   help="override the model dir (reference.json's is provenance; "
                        "its absolute path won't exist on another machine)")
    p.add_argument("--quick", action="store_true",
                   help="--tokens: print the prefix diagnostics and skip the accuracy "
                        "tier (heavy deps). Grades nothing; exits 2.")
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


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def score_tokens(args, model_dir, comps, thr):
    """Score a generated-ids file (getp output) against the golden completions.

    One line of space-separated ids per request, in request order. The engine may
    generate more than the golden `max_new` (e.g. the getp default steps=128); the
    surplus is ignored, so one timed getp run scores without a second, shorter run.
    """
    # Not read_id_lines(): a request that generated nothing writes an empty line,
    # and dropping it would misreport the failure as a line-count mismatch.
    raw = open(args.tokens).read().split("\n")
    if raw and raw[-1] == "":
        raw.pop()
    gen = [[int(x) for x in ln.split()] for ln in raw]
    if len(gen) != len(comps):
        sys.exit(f"{args.tokens}: {len(gen)} lines != {len(comps)} requests in the dataset")

    print("  gate: " + ("(none -- --quick skips the accuracy tier)" if args.quick else
                        f"meteor>={thr['meteor']}  bertscore_f1>={thr['bertscore_f1']}"),
          flush=True)

    fracs, n_exact = [], 0
    for i, (g, c) in enumerate(zip(gen, comps)):
        pre = common_prefix(g, c)
        frac = pre / len(c)
        fracs.append(frac)
        exact = pre == len(c)
        n_exact += exact
        div = "" if exact else f"  first diff @{pre}: gold {c[pre]} != gen " + (
            str(g[pre]) if pre < len(g) else "<end>")
        print(f"  [{i}] gen={len(g):4d} gold={len(c):4d}  "
              f"prefix={pre:4d}/{len(c)} ({frac*100:6.2f}%){div}", flush=True)

    # Prefix statistics are DIAGNOSTIC, not a gate: the announced accuracy gate is
    # METEOR/BERTScore, and an engine may legitimately diverge from the fp32
    # reference (bf16/fp8 weights or KV cache) while still being correct. Read
    # these to tell a numerically-different engine from a broken one; the
    # thresholds they cite are advisory reference points.
    mean_prefix = sum(fracs) / len(fracs)
    bad = [i for i, f in enumerate(fracs) if f < thr["getp_min_prefix"]]
    print()
    print(f"  requests matching exactly = {n_exact}/{len(comps)}")
    print(f"  mean prefix agreement     = {mean_prefix*100:.3f}%        [diagnostic]")
    print(f"  worst request             = {min(fracs)*100:.3f}%        [diagnostic]")
    print(f"  below {thr['getp_min_prefix']*100:.0f}% floor           = {len(bad)}/{len(fracs)}"
          f" ({len(bad)/len(fracs)*100:.2f}%)        [diagnostic]"
          + (f"  reqs {bad[:8]}{'...' if len(bad) > 8 else ''}" if bad else ""))

    if args.quick:
        return None                      # nothing was gated
    preds = [g[:len(c)] for g, c in zip(gen, comps)]
    return score_fuzzy(model_dir, preds, comps, thr)


def main():
    import math
    args = parse_args()
    data = args.dir or os.path.join(_HERE, args.model)
    ref = json.load(open(os.path.join(data, "reference.json")))
    model_dir = args.model_dir or ref["model_dir"]
    thr = json.load(open(args.thresholds))
    prompts = read_id_lines(os.path.join(data, "prompts.i32.txt"))
    comps = read_id_lines(os.path.join(data, "completions.i32.txt"))
    recs = ref["requests"]
    n = len(recs)
    assert len(prompts) == len(comps) == n, "dataset length mismatch"

    if args.tokens:
        print(f"[{args.model}] {args.tokens}  ({n} requests)", flush=True)
        ok = score_tokens(args, model_dir, comps, thr)
        print()
        if ok is None:      # --quick: diagnostics only, so say so rather than pass
            print("  RESULT: not graded (--quick skipped the accuracy gate)")
            sys.exit(2)
        print("  RESULT:", "ok" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    if not os.path.exists(args.run):
        sys.exit(f"C run binary not found: {args.run} (build with `make run`)")
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


def score_fuzzy(model_dir, pred_ids, ref_ids, thr):
    """Detokenize both sides and score METEOR + BERTScore. Heavy deps."""
    from transformers import AutoTokenizer
    import evaluate  # noqa
    tok = AutoTokenizer.from_pretrained(model_dir)
    preds = [tok.decode(ids) for ids in pred_ids]
    refs = [tok.decode(ids) for ids in ref_ids]
    meteor = evaluate.load("meteor").compute(predictions=preds, references=refs)["meteor"]
    # BERTScore's tokenizer raises on an empty/whitespace-only prediction, which a
    # request that generated nothing produces. Score those 0 (maximally wrong) and
    # keep them out of the compute call, rather than crashing the gate.
    live = [i for i, p in enumerate(preds) if p.strip()]
    f1s = [0.0] * len(preds)
    if live:
        bs = evaluate.load("bertscore").compute(
            predictions=[preds[i] for i in live],
            references=[refs[i] for i in live], lang="en")
        for i, v in zip(live, bs["f1"]):
            f1s[i] = v
    f1 = sum(f1s) / len(f1s)
    empty = len(preds) - len(live)
    print(f"  METEOR = {meteor:.4f}  BERTScore-F1 = {f1:.4f}"
          + (f"  ({empty} empty prediction(s) scored 0)" if empty else ""))
    return meteor >= thr["meteor"] and f1 >= thr["bertscore_f1"]


def run_fuzzy(args, model_dir, prompts, comps, recs, thr):
    """Free-run greedy generation vs golden completion, scored METEOR+BERTScore."""
    preds = []
    for pids, rec in zip(prompts, recs):
        max_new = args.max_new or rec["completion_len"]
        out = run_c(args.run, model_dir, pids, "gen", max_new)
        gen_ids = []
        for line in out.splitlines():
            if line.startswith("completion"):
                gen_ids = [int(x) for x in line.split()[1:]]
        preds.append(gen_ids)
    return score_fuzzy(model_dir, preds, comps, thr)


if __name__ == "__main__":
    main()
