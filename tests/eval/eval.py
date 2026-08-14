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
  reference, so sameness cannot gate. Read the prefix numbers anyway: agreement
  that collapses while throughput jumps is the sign that something broke.

Thresholds for both come from threshold.json.

Exit codes: 0 = ok, 1 = the gate failed, 2 = not graded (nothing was scored),
3 = environment fault (missing deps, cold cache, no network). A grading script
must not record 3 as a candidate failure.

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
    p.add_argument("--steps", type=int, default=None,
                   help="--tokens: the STEPS the run requested, so a generation "
                        "capped below the reference length is reported as a "
                        "misconfiguration rather than graded as a failure")
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


_MODEL_HINT = {"dsv2lite": "deepseek", "glm47": "glm"}

# Warnings are re-printed next to RESULT: a caution that scrolls past 512
# diagnostic lines is a caution nobody reads, and the cases that raise one
# (wrong --model-dir, generation shorter than the reference) produce a
# plausible-looking verdict rather than an obvious failure.
_WARNINGS = []


def warn(msg):
    _WARNINGS.append(msg)
    print(f"  WARNING: {msg}", flush=True)


def print_warnings():
    """Re-print next to the verdict. Must run before EVERY exit path, including
    the early ones -- a warning 500 lines up is a warning nobody reads."""
    for w in _WARNINGS:
        print(f"  WARNING: {w}")


def check_dataset_model(args, ref, data):
    """Fail fast when MODEL, DATA and --model-dir do not describe the same model.

    They are independent variables in the Makefile, so `MODEL=glm47
    DATA=<dsv2lite dir>` silently scores GLM output against DeepSeek ids with the
    GLM tokenizer and reports a plain FAIL. Not every dataset carries the same
    provenance keys -- the in-repo dev sets have "model", the published set has
    only "model_dir" and a manifest -- so cross-check whatever is present.
    """
    stated = ref.get("model")
    if stated is None:
        man = os.path.join(data, "manifest.json")
        if os.path.exists(man):
            stated = json.load(open(man)).get("model")
    if stated and stated != args.model:
        print_warnings()
        print(f"dataset {data} is for model '{stated}', but MODEL={args.model}",
              file=sys.stderr)
        sys.exit(2)                      # misconfiguration, not a gate failure

    # model_dir is provenance -- the weight-directory name on the machine that
    # generated the set -- so a rename there must not block a consistent dataset.
    # Heuristic, therefore a warning; the authoritative check above stays fatal.
    ref_dir = os.path.basename(str(ref.get("model_dir", "")).rstrip("/")).lower()
    hint = _MODEL_HINT.get(args.model)
    if ref_dir and hint and hint not in ref_dir and not stated:
        warn(f"dataset {data} was generated from '{ref['model_dir']}', which does "
             f"not look like a {args.model} model")
    if args.model_dir and ref_dir:
        got = os.path.basename(args.model_dir.rstrip("/")).lower()
        if got != ref_dir:
            warn(f"--model-dir is '{got}' but the dataset was generated from "
                 f"'{ref_dir}' -- the engine may be running the wrong weights")


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
    try:
        raw = open(args.tokens).read().split("\n")
    except OSError as e:
        print_warnings()
        print(f"cannot read {args.tokens}: {e}", file=sys.stderr)
        sys.exit(2)                      # misconfiguration, not a gate failure
    
    if raw and raw[-1] == "":
        raw.pop()
    gen = [[int(x) for x in ln.split()] for ln in raw]
    if len(gen) != len(comps):
        print_warnings()
        print(f"{args.tokens}: {len(gen)} lines != {len(comps)} requests in the dataset",
              file=sys.stderr)
        sys.exit(2)                      # misconfiguration, not a gate failure

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
    print(f"  mean prefix agreement     = {mean_prefix*100:.3f}%        [diagnostic, "
          f"advisory floor {thr['getp_prefix']*100:.0f}%]")
    print(f"  worst request             = {min(fracs)*100:.3f}%        [diagnostic]")
    print(f"  below {thr['getp_min_prefix']*100:.0f}% floor           = {len(bad)}/{len(fracs)}"
          f" ({len(bad)/len(fracs)*100:.2f}%)        [diagnostic]"
          + (f"  reqs {bad[:8]}{'...' if len(bad) > 8 else ''}" if bad else ""))

    # Truncating over-long generations is free, but a generation SHORTER than the
    # reference silently costs recall on every metric. The usual cause is STEPS
    # capped below the reference completion length, which fails a correct engine.
    short = [i for i, (g, c) in enumerate(zip(gen, comps)) if len(g) < len(c)]
    if short:
        worst = min((len(gen[i]) - len(comps[i]), i) for i in short)[1]
        print(f"  shorter than reference    = {len(short)}/{len(comps)}"
              f"        [diagnostic]  worst: req{worst} "
              f"{len(gen[worst])} vs {len(comps[worst])} tokens")
        # Every short generation the same length is a hard cap (STEPS below the
        # reference length), not an engine defect. Grading that would report FAIL
        # for a correct engine, so refuse to grade instead of scoring it.
        # An engine that stops on EOS gives mixed short lengths, so "all equal"
        # was too strict -- one early stop let a capped run through to a FAIL.
        # Prefer the cap the caller actually asked for; fall back to the most
        # common short length when --steps was not passed.
        from collections import Counter
        counts = Counter(len(gen[i]) for i in short)
        # --steps is a HINT, not a switch: an engine whose own loop stops one token
        # early would otherwise match nothing and escape the check entirely.
        cap, n_at_cap = counts.most_common(1)[0]
        # Prefer the requested cap only when it explains at least as many requests
        # as the measured one; a couple of requests ending exactly at STEPS must
        # not hide a larger cap somewhere else.
        if args.steps and counts.get(args.steps, 0) >= n_at_cap:
            cap, n_at_cap = args.steps, counts[args.steps]
        longest = max(len(c) for c in comps)
        # cap 0 is not reachable (getp_eval.c floors steps<=0 at GETP_DEFAULT_STEPS),
        # so an engine emitting nothing is a real failure, not a misconfiguration.
        # If STEPS is at least the longest reference, no cap can explain a short
        # generation: the engine truncated on its own, which is a real defect and
        # must reach the gate rather than be excused as a misconfiguration.
        capped = (args.steps or 0) < longest
        if capped and cap > 0 and n_at_cap >= max(2, thr["getp_bad_frac"] * len(comps)) \
                and cap < longest:
            print_warnings()
            # exit 2 = "not graded", same as --quick; 1 is reserved for a real FAIL
            print(f"\n  RESULT: not graded -- {n_at_cap}/{len(comps)} generations "
                  f"stop at exactly {cap} tokens, below the reference "
                  f"(max {longest}).\n"
                  f"  That is a generation cap, not an engine defect: re-run with "
                  f"STEPS >= {longest}.", flush=True)
            sys.exit(2)
        if capped:
            warn(f"{len(short)}/{len(comps)} requests generated fewer tokens than the "
                 f"reference; if STEPS caps generation below the reference length "
                 f"(max {longest}), the gate will fail a correct engine")
        else:
            warn(f"{len(short)}/{len(comps)} requests generated fewer tokens than the "
                 f"reference, and STEPS={args.steps} cannot be the cause (>= the "
                 f"longest reference, {longest}) -- the engine stopped early itself")

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
    # Not an assert: `python -O` would drop it and zip() would then hide the
    # difference silently. A malformed dataset is a misconfiguration, not a FAIL.
    if not (len(prompts) == len(comps) == n):
        print(f"dataset {data} is inconsistent: {len(prompts)} prompts, "
              f"{len(comps)} completions, {n} reference records", file=sys.stderr)
        sys.exit(2)
    check_dataset_model(args, ref, data)

    if args.tokens:
        print(f"[{args.model}] {args.tokens}  ({n} requests)", flush=True)
        ok = score_tokens(args, model_dir, comps, thr)
        print()
        print_warnings()
        if ok is None:      # --quick: diagnostics only, so say so rather than pass
            print("  RESULT: not graded (--quick skipped the accuracy gate)")
            sys.exit(2)
        print("  RESULT:", "ok" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    if not os.path.exists(args.run):
        print_warnings()
        print(f"C run binary not found: {args.run} (build with `make run`)",
              file=sys.stderr)
        sys.exit(3)                      # environment fault, not a gate failure
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
    print_warnings()
    print("  RESULT:", "ok" if ok else "FAIL")
    sys.exit(0 if ok else 1)


_WARM = ("run `make eval-warm` once on a networked machine to fill the metric, "
         "nltk and roberta-large caches; HF_HUB_OFFLINE=1 works after that")


def score_fuzzy(model_dir, pred_ids, ref_ids, thr):
    """Detokenize both sides and score METEOR + BERTScore. Heavy deps."""
    from transformers import AutoTokenizer
    try:
        import evaluate  # noqa
    except ModuleNotFoundError:
        print_warnings()
        print("the accuracy tier needs the `fuzzy` extra: `uv sync --extra fuzzy`",
              file=sys.stderr)
        sys.exit(3)
    # model_dir may be a Hub repo id (the published reference.json records one),
    # so this load can hit the network too -- same guard, same hint.
    try:
        tok = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        print_warnings()
        print(f"could not load the tokenizer for '{model_dir}' "
              f"({type(e).__name__}: {e})\n  If this machine is offline or the "
              f"caches are cold, {_WARM}.", file=sys.stderr)
        sys.exit(3)
    preds = [tok.decode(ids) for ids in pred_ids]
    refs = [tok.decode(ids) for ids in ref_ids]
    try:
        meteor = evaluate.load("meteor").compute(predictions=preds, references=refs)["meteor"]
    except Exception as e:
        print_warnings()
        print(f"could not load or run the METEOR metric ({type(e).__name__}: {e})\n"
              f"  If this machine is offline or the caches are cold, {_WARM}.",
              file=sys.stderr)
        sys.exit(3)
    # BERTScore's tokenizer raises on an empty/whitespace-only prediction, which a
    # request that generated nothing produces. Score those 0 (maximally wrong) and
    # keep them out of the compute call, rather than crashing the gate.
    live = [i for i, p in enumerate(preds) if p.strip()]
    f1s = [0.0] * len(preds)
    if live:
        try:
            bs = evaluate.load("bertscore").compute(
                predictions=[preds[i] for i in live],
                references=[refs[i] for i in live], lang="en")
        except Exception as e:
            print_warnings()
            print(f"could not load or run BERTScore ({type(e).__name__}: {e})\n"
                  f"  If this machine is offline or the caches are cold, {_WARM}.",
                  file=sys.stderr)
            sys.exit(3)
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
