"""Cross-check the C tokenizer (tok_cli) against HuggingFace on ASCII text.

  uv run python tests/tokenizer/compare_hf.py <model_dir> <tok_cli_path>

Compares encode (add_special_tokens=False) and decode round-trip on a panel of
ASCII/English strings. Non-ASCII is intentionally out of scope (see tokenizer.h).
"""
import subprocess, sys
from transformers import AutoTokenizer

PANEL = [
    "The capital of France is",
    "Hello, world! 123 + 456 = 579.",
    "def add(a, b):\n    return a + b\n",
    "   leading and trailing spaces   ",
    "Multiple    spaces\tand\ttabs",
    "Line one\nLine two\n\nLine four",
    "Punctuation?! (yes) -- it's [tricky]; really...",
    "CamelCase snake_case kebab-case SCREAMING_CASE",
    "Numbers: 0 1 42 1000 3.14159 0x1F 1,000,000",
    "URL https://example.com/path?q=1&r=2#frag",
    "Quotes: 'single' \"double\" `back`",
    "",
    "a",
    "    ",
]

def c_encode(model_dir, tok_cli, text):
    # tok_cli prints: "<ids>\n<decoded><\n>". Decoded text may itself contain
    # newlines, so split only on the first one and drop the single trailing \n.
    out = subprocess.run([tok_cli, model_dir, text], capture_output=True, text=True)
    head, _, rest = out.stdout.partition("\n")
    ids = [int(x) for x in head.split()] if head.strip() else []
    decoded = rest[:-1] if rest.endswith("\n") else rest
    return ids, decoded

def main():
    model_dir, tok_cli = sys.argv[1], sys.argv[2]
    hf = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    enc_fail = dec_fail = 0
    for s in PANEL:
        hf_ids = hf.encode(s, add_special_tokens=False)
        c_ids, c_dec = c_encode(model_dir, tok_cli, s)
        enc_ok = hf_ids == c_ids
        dec_ok = c_dec == s
        if not enc_ok: enc_fail += 1
        if not dec_ok: dec_fail += 1
        tag = "ok " if (enc_ok and dec_ok) else "DIFF"
        print(f"[{tag}] {s!r}")
        if not enc_ok:
            print(f"      HF : {hf_ids}")
            print(f"      C  : {c_ids}")
        if not dec_ok:
            print(f"      roundtrip != input: {c_dec!r}")
    print(f"\nencode mismatches: {enc_fail}/{len(PANEL)}   "
          f"decode mismatches: {dec_fail}/{len(PANEL)}")
    sys.exit(1 if (enc_fail or dec_fail) else 0)

if __name__ == "__main__":
    main()
