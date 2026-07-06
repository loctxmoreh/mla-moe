# MLA-MoE

An educational, from-scratch C inference engine for transformer models that use
Multi-head Latent Attention (MLA) and Mixture-of-Experts (MoE) — the
DeepSeek-V2 architecture family. In the spirit of Andrej Karpathy's llama2.c:
one readable C codebase where every operation is explicit and auditable.

**Targets:** DeepSeek-V2-Lite (`deepseek_v2`) and GLM-4.7-Flash (`glm4_moe_lite`).

## Build

The Makefile defaults `CC` to `/opt/rocm/llvm/bin/clang`:

```sh
make                 # optimized; builds `run` (engine) and `mla-moe` (weight inspector)
make DUMP=1          # + -DMLA_ENABLE_DUMP: writes oracle-named intermediates for validation
```

Default builds carry no dump code (it is `#ifdef`-gated). **Run `make clean`
when toggling `DUMP=1`** — make does not rebuild on a CFLAGS-only change.

Debug builds (ASan + UBSan): `make debug-run` / `make debug-tool`.

## Run

```
run <model_dir> [tokens.i32.bin | -p "text"] [dump_dir|ppl]
```

`<model_dir>` holds `config.json`, `model.safetensors.index.json`, and the
shards; the model family is auto-detected from `config.json`. `tokens.i32.bin`
is raw little-endian int32 token ids (the oracle's `input_ids.i32.bin` files are
exactly this). Alternatively, `-p "text"` tokenizes a prompt with the in-C
tokenizer (reads `tokenizer.json`; pre-tokenizer chosen from `config.json`), and
generated ids are decoded back to text.

```sh
DSV=/path/to/deepseek-ai/DeepSeek-V2-Lite
GLM=/path/to/zai-org/GLM-4.7-Flash

# weight-load smoke test (no tokens)
./run "$DSV"

# prefill + greedy absorbed-decode; prints generated token ids
./run "$DSV" tests/oracle/dumps/dsv2lite/input_ids.i32.bin

# same, but from a text prompt (tokenized + detokenized in C)
./run "$DSV" -p "The capital of France is"
./run "$GLM" -p "The capital of France is"

# (DUMP=1 build) prefill + one decode step, writing intermediates to <dump_dir>
./run "$GLM" tests/oracle/dumps/glm47/input_ids.i32.bin /tmp/cdump
```

## Tokenizer

`-p "text"` drives an in-C byte-level BPE tokenizer (GPT-2 family) that reads
the model's `tokenizer.json` (vocab + merges + added tokens) and `config.json`
(bos/eos, model family).  Prompts are encoded with **no BOS**, matching the
oracle's tokenization and the validated reference path.

It is **faithful to HuggingFace for ASCII/English text**. The pre-tokenizer's
non-ASCII letter/punctuation/CJK handling is intentionally omitted, so non-ASCII
input still yields valid tokens but may split differently from HF.

Cross-check the C tokenizer against HuggingFace on an ASCII panel (no model
weights loaded):

```sh
make tok-cli   # builds tests/tokenizer/tok_cli
uv run python tests/tokenizer/compare_hf.py "$DSV" ./tests/tokenizer/tok_cli
uv run python tests/tokenizer/compare_hf.py "$GLM" ./tests/tokenizer/tok_cli
```

## Validate against the oracle

The oracle (`tests/oracle/`) dumps ground-truth activations from transformers in
fp32. Run a `DUMP=1` build, then diff:

```sh
make clean && make DUMP=1
./run "$DSV" tests/oracle/dumps/dsv2lite/input_ids.i32.bin /tmp/cdump
cd tests/oracle
uv run python compare_prefill.py /tmp/cdump dsv2lite   # 0 failures; argmax 8913 (" Paris")
uv run python compare_decode.py  /tmp/cdump dsv2lite
```

Use `glm47` + `$GLM` for the other model (argmax 12089). Regenerate the dumps
with `uv run python tests/oracle/gen_oracle.py <dsv2lite|glm47>` (see
`tests/oracle/README.md`; glm47 needs ~125 GB RAM).

## Correctness eval

The primary correctness gate scores the engine against a **frozen golden dataset**
of greedy sequences generated once from HF (fp32), not per-tensor dumps. Three
device-neutral metrics: teacher-forced top-1 agreement, perplexity relative error,
and (optional) free-run METEOR/BERTScore.

```sh
# 1. generate the golden dataset from HF (greedy, fp32); run once per model
make eval-gen MODEL=dsv2lite        # writes tests/eval/dsv2lite/{prompts,completions}.i32.txt, reference.json

# 2. score the C engine (teacher-forced top-1 both paths + ppl rel-err)
make eval MODEL=dsv2lite            # add FUZZY=1 for the METEOR/BERTScore tier
```

The engine exposes the eval modes directly:

```sh
./run "$DSV" seq.i32.bin teacher 5  # teacher-forced top-1: 'P'/'D' rows, decode region pos>=5
./run "$DSV" prompt.i32.bin gen 64  # greedy free-run; prints 'completion <ids>'
```

For the GPU/HIP port, freeze the validated CPU engine as a **tagged, buildable**
reference (not the mutating `run.c`): `git tag cpu-oracle-v1 && make ref-binary`
builds `run-ref` from that tag to diff kernels against. The per-tensor oracle
(`tests/oracle/`, below) is retained as the bring-up microscope for localizing a
failing kernel.

## Limitations

Single stream (batch=1), greedy sampling, ASCII/English tokenizer (see above).
Performance work (blocked matmuls, threading) is deferred — correctness first.
