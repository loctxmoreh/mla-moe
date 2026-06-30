"""Compare the C engine's teacher-forced perplexity against an HF reference.

    uv run python compare_ppl.py [dsv2lite|glm47] [options]

Tokenizes a text with the model's HF tokenizer, computes the HF teacher-forced
perplexity (fp32, eager — matching the C engine's bf16-weights/fp32-compute
contract), runs the C engine in `ppl` mode on the SAME token ids, and asserts
the relative PPL error is within --tol. Identical ids go to both engines, so
tokenization/BOS choices cannot desync them.

Options:
  -t, --text   STR   text to score (default: a short prose paragraph)
  -f, --file   PATH  read the text from a file instead
  -r, --run    PATH  path to the C `run` binary (default: <repo>/run)
  --tol        FLOAT relative PPL error threshold (default: 0.01 = 1%)
  --max-tokens INT   cap sequence length (default 4096; the C buffer limit)
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import load_oracle

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))

DEFAULT_TEXT = (
    "The history of computing is a story of abstraction. Each generation of "
    "engineers built tools that hid the complexity of the layer beneath, so the "
    "next generation could reason about larger ideas without drowning in detail."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default="dsv2lite", choices=["dsv2lite", "glm47"])
    p.add_argument("-t", "--text", default=None)
    p.add_argument("-f", "--file", default=None)
    p.add_argument("-r", "--run", default=os.path.join(_REPO, "run"))
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--max-tokens", type=int, default=4096)
    return p.parse_args()


def hf_perplexity(model, ids):
    """Mean teacher-forced NLL -> exp, over positions [0, n-1): p predicts ids[p+1]."""
    with torch.no_grad():
        logits = model(ids).logits[0].float()        # [seq, vocab], fp32
    shift_logits = logits[:-1, :]
    shift_targets = ids[0, 1:]
    nll = torch.nn.functional.cross_entropy(shift_logits, shift_targets, reduction="sum")
    ntok = shift_targets.numel()
    return float(torch.exp(nll / ntok)), float(nll), int(ntok)


def c_perplexity(run_bin, model_dir, ids):
    """Write ids to a temp .i32.bin, run the C engine in ppl mode, parse 'ppl <v>'."""
    with tempfile.NamedTemporaryFile(suffix=".i32.bin", delete=False) as f:
        np.asarray(ids[0].tolist(), dtype="<i4").tofile(f)
        tokens_path = f.name
    try:
        out = subprocess.run(
            [run_bin, model_dir, tokens_path, "ppl"],
            capture_output=True, text=True, check=True,
        ).stdout
    finally:
        os.unlink(tokens_path)
    for line in out.splitlines():
        if line.startswith("ppl "):
            return float(line.split()[1]), out
    raise RuntimeError(f"no 'ppl' line in C engine output:\n{out}")


def main():
    args = parse_args()
    if not os.path.exists(args.run):
        sys.exit(f"C run binary not found: {args.run} (build with `make CC=gcc run`)")

    model_dir = load_oracle.manifest(os.path.join(_HERE, "dumps", args.model))["model_dir"]
    text = open(args.file).read() if args.file else (args.text or DEFAULT_TEXT)

    print(f"[{args.model}] {model_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(text, return_tensors="pt").input_ids
    if ids.shape[1] > args.max_tokens:
        ids = ids[:, : args.max_tokens]                # truncate BOTH engines identically
        print(f"  truncated to {args.max_tokens} tokens", flush=True)
    seq = ids.shape[1]
    if seq < 2:
        sys.exit("need >= 2 tokens to score perplexity")
    print(f"  tokens={seq}", flush=True)

    print("  loading HF model (fp32, eager) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True)
    model.eval()

    hf_ppl, hf_nll, ntok = hf_perplexity(model, ids)
    c_ppl, _ = c_perplexity(args.run, model_dir, ids)

    rel = abs(c_ppl - hf_ppl) / hf_ppl
    print()
    print(f"  HF  perplexity = {hf_ppl:.6f}  (sum_nll={hf_nll:.6f}, {ntok} targets)")
    print(f"  C   perplexity = {c_ppl:.6f}")
    print(f"  rel_err = {rel:.3e}   tol = {args.tol:.1e}   "
          f"{'ok' if rel <= args.tol else 'FAIL'}")
    sys.exit(0 if rel <= args.tol else 1)


if __name__ == "__main__":
    main()
