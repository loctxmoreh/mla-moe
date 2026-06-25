# Oracle — ground-truth activations for the MLA+MoE models

Reference activations for a single forward pass, used to validate the C engine's
`forward()`/decode once implemented. The reference is transformers' **native**
model (transformers 5.x) run in **fp32** — matching the C contract of bf16
weights / fp32 compute.

Two models, each with its own `dumps/<model>/`:

| key        | model            | query     | RoPE              | router (MoE)            |
|------------|------------------|-----------|-------------------|-------------------------|
| `dsv2lite` | DeepSeek-V2-Lite | `q_proj`  | complex           | softmax + greedy top-k  |
| `glm47`    | GLM-4.7-Flash    | q-LoRA 768| interleaved       | `noaux_tc` (sigmoid + e_score bias, group, norm, ×1.8) |

Two phases each, mirroring the C engine's two code paths:

| Phase   | q_len | C engine path        | What it dumps |
|---------|-------|----------------------|---------------|
| prefill | N     | **unabsorbed** MLA   | logits, per-layer attn/mlp out, hidden states, layer-0 attn internals, first-MoE-layer router |
| decode  | 1     | **absorbed** MLA     | decode logits, layer-0 latent cache + absorbed intermediates |

## Validation method

Two reference implementations with distinct roles:

- **HF `model`** (`AutoModelForCausalLM`, fp32, eager) is the authoritative
  end-to-end ground truth — the dumped `*_logits`, `*_hidden_states`, and per-layer
  `attn_out`/`mlp_out` come from its real forward pass.
- **`mla_reference`** holds the two explicit MLA algorithms the C engine ports
  (unabsorbed + absorbed), validated against HF. 
  - It exists because stock transformers has no absorbed path (it always
    decompresses `kv_b_proj`, even at decode) and does not expose per-stage
    intermediates.

Guarantees (asserted, both models, fp32 throughout): prefill unabsorbed == HF
bit-for-bit; decode absorbed == unabsorbed == HF ≤1e-7. The dumped MLA intermediates
are therefore HF-validated. This cross-check surfaced the `kv_b` head-interleave
layout. Sanity: both models complete "The capital of France is" → " Paris".

## Regenerate

```sh
cd tests/oracle
uv run python gen_oracle.py dsv2lite   # ~63GB RAM
uv run python gen_oracle.py glm47       # ~125GB RAM, 48 shards
# (no arg defaults to dsv2lite; pass both to do both)
```

## Consume

Dumps are raw **C-order little-endian** (`.f32.bin` = float32, `.i32.bin` =
int32); shapes/dtypes are in each `dumps/<model>/manifest.json`. From Python:
`load_oracle.load("decode_logits", "dumps/glm47")`. From C: `fread` into a
row-major buffer of the manifest shape.

## Spec notes the C engine must honor (per-model `manifest.json` `notes`)

- **kv_b_proj split is head-INTERLEAVED** (both models): `[k_nope(qk_nope) |
  value(v_head_dim)]` per head, *not* `[all k_nope][all value]`. `layer0_W_UK`/
  `layer0_W_UV` are the correct split, per-head stride `(qk_nope+v_head_dim)*kv_lora`.
  (`src/run.c` splits this way as of the kv_b fix. GLM has unequal blocks: 192/256.)
- **RoPE on the `qk_rope_head_dim` slice** — but the flavor differs:
  `dsv2lite` = complex (adjacent-pair) rope; `glm47` = interleaved rope
  (`apply_rotary_pos_emb_interleave`: rotate even/odd pairs, output split-half).
  Softmax uses plain `scaling = q_head_dim**-0.5` for both (no mscale here).
- **Decode cache** stores `c_kv (kv_lora_rank) || k_pe (qk_rope_head_dim)` per
  position, matching `RunState.kv_cache`.
- **GLM query** uses a LoRA: `q = q_b_proj(q_a_layernorm(q_a_proj(x)))`; DSV2 uses
  a single `q_proj`. RMSNorm eps differs (dsv2 1e-6, glm 1e-5).
