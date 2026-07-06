"""Device-agnostic prefill/decode performance benchmark.

    uv run python bench.py [dsv2lite|glm47] [options]

Drives the C engine's `bench` mode (which owns the wall-clock timing) across a
sweep of prefill lengths and reports two regimes separately:

  - prefill tok/s  (compute-bound dense GEMM; prefill_ms == time-to-first-token)
  - decode  tok/s + TPOT (memory-bound, per-token MoE routing)

Device-agnostic: the timing lives inside `run` around the forward calls, whose
host logits force any backend to synchronize at the clock boundary. So the SAME
harness measures a CPU build, `run-ref`, or a future GPU build -- just point
`-r` at the binary. `--compare baseline.json` prints per-regime speedup, the
perf analogue of the `run` vs `run-ref` correctness diff.

Options:
  -r, --run PATH     C run binary (default <repo>/run)
  --model-dir PATH   override model dir (default from <model>/reference.json)
  --prefill LIST     comma-separated prefill lengths (default 128,512,1024,2048)
  --decode INT       decode steps per run (default 64)
  --reps INT         timed reps; rep 0 is warmup, excluded (default 5)
  -o, --out PATH     write results JSON here
  --compare PATH     print speedup vs a prior results JSON (same prefill lengths)
"""
import argparse
import datetime
import json
import os
import platform
import struct
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default="dsv2lite", choices=["dsv2lite", "glm47"])
    p.add_argument("-r", "--run", default=os.path.join(_REPO, "run"))
    p.add_argument("--model-dir", default=None)
    p.add_argument("--prefill", default="128,512,1024,2048")
    p.add_argument("--decode", type=int, default=64)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--compare", default=None)
    return p.parse_args()


def read_id_lines(path):
    with open(path) as f:
        return [[int(x) for x in ln.split()] for ln in f if ln.strip()]


def write_i32(ids):
    f = tempfile.NamedTemporaryFile(suffix=".i32.bin", delete=False)
    f.write(struct.pack("<%di" % len(ids), *ids))
    f.close()
    return f.name


def make_seq(base, n):
    """A length-n token sequence (content is irrelevant to timing): tile+truncate."""
    out = list(base)
    while len(out) < n:
        out += base
    return out[:n]


def parse_kv(line):
    """'<tag> k1=v1 k2=v2 ...' -> {k1: v1, ...} (values as float where possible)."""
    d = {}
    for tok in line.split()[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            d[k] = float(v)
        except ValueError:
            d[k] = v
    return d


def run_bench(run_bin, model_dir, ids, decode, reps):
    """-> (summary_dict, config_dict) parsed from the C engine's stdout."""
    path = write_i32(ids)
    try:
        out = subprocess.run([run_bin, model_dir, path, "bench", str(decode), str(reps)],
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(path)
    summary, config = None, {}
    for line in out.splitlines():
        if line.startswith("bench_summary "):
            summary = parse_kv(line)
        elif line.startswith("Config:"):
            config = parse_kv(line)
    if summary is None:
        sys.exit("no bench_summary line in engine output:\n" + out)
    return summary, config


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def git_commit():
    try:
        return subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main():
    args = parse_args()
    if not os.path.exists(args.run):
        sys.exit(f"C run binary not found: {args.run} (build with `make run`)")
    ref = json.load(open(os.path.join(_HERE, "..", "eval", args.model, "reference.json")))
    model_dir = args.model_dir or ref["model_dir"]
    prompts = read_id_lines(os.path.join(_HERE, "..", "eval", args.model, "prompts.i32.txt"))
    base = max(prompts, key=len)   # tile the longest golden prompt to any length

    prefills = [int(x) for x in args.prefill.split(",") if x.strip()]
    over = [p for p in prefills if p + args.decode > 4096]
    if over:
        print(f"note: dropping prefill lengths {over} -- prefill+decode > 4096 "
              f"(run.c reads a fixed int tokens[4096])", flush=True)
        prefills = [p for p in prefills if p + args.decode <= 4096]

    print(f"[{args.model}] {model_dir}", flush=True)
    print(f"  run={args.run}  decode={args.decode}  reps={args.reps} (rep 0 warmup)", flush=True)
    print(f"  {'prefill':>8} {'prefill_ms':>11} {'prefill tok/s':>14} "
          f"{'decode tok/s':>13} {'TPOT ms':>9}", flush=True)

    rows, config = [], {}
    for p in prefills:
        s, config = run_bench(args.run, model_dir, make_seq(base, p), args.decode, args.reps)
        rows.append({
            "prefill_len": p,
            "prefill_ms": s["prefill_ms_median"],
            "prefill_tok_s": s["prefill_tok_s_median"],
            "decode_tok_s": s["decode_tok_s_median"],
            "tpot_ms": s["tpot_ms_median"],
        })
        print(f"  {p:>8} {s['prefill_ms_median']:>11.1f} {s['prefill_tok_s_median']:>14.2f} "
              f"{s['decode_tok_s_median']:>13.2f} {s['tpot_ms_median']:>9.2f}", flush=True)

    result = {
        "model": args.model,
        "model_dir": model_dir,
        "run_bin": os.path.abspath(args.run),
        "git_commit": git_commit(),
        "compiler": "clang -O2 (see Makefile)",
        "decode": args.decode,
        "reps": args.reps,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "machine": {
            "uname": list(platform.uname()),
            "cpu": cpu_model(),
        },
        "config": {k: int(config[k]) for k in ("n_layers", "hidden", "vocab") if k in config},
        "rows": rows,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  wrote {args.out}", flush=True)

    if args.compare:
        compare(result, json.load(open(args.compare)))


def compare(cur, base):
    """Per-regime speedup vs a baseline results JSON, matched by prefill length."""
    b = {r["prefill_len"]: r for r in base["rows"]}
    print(f"\n  speedup vs {base.get('run_bin','?')} @ {base.get('git_commit','?')[:8]}", flush=True)
    print(f"  {'prefill':>8} {'prefill x':>11} {'decode x':>10}", flush=True)
    for r in cur["rows"]:
        o = b.get(r["prefill_len"])
        if not o:
            continue
        pf = r["prefill_tok_s"] / o["prefill_tok_s"] if o["prefill_tok_s"] else float("nan")
        dc = r["decode_tok_s"] / o["decode_tok_s"] if o["decode_tok_s"] else float("nan")
        print(f"  {r['prefill_len']:>8} {pf:>10.2f}x {dc:>9.2f}x", flush=True)


if __name__ == "__main__":
    main()
