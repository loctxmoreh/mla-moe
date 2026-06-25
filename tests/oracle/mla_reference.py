"""Standalone MLA reference: unabsorbed (prefill) and absorbed (decode) forms.

The two forms are mathematically equivalent and define the spec the C engine
targets. Tensors are fp32 derived from the bf16 checkpoint (bf16 weights, fp32
compute). gen_oracle.py asserts both forms agree with each other and with the HF
reference, then dumps the intermediates.
"""

import torch


def rms_norm(x, weight, eps=1e-6):
    """RMS-normalizes the last axis.

    Math (numpy):
        x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * weight

    Args:
        x: Input array; normalized over its last axis.
        weight: Per-channel scale, shape [x.shape[-1]].
        eps: Variance floor.

    Returns:
        Array of the same shape as `x`.
    """
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(var + eps))


def apply_rope(x, freqs_cis):
    """Applies DeepSeek complex (interleaved-pair) RoPE to the last axis.

    Adjacent dims form complex pairs rotated by `freqs_cis`; matches transformers'
    deepseek_v2 `apply_rotary_emb`. The YaRN mscale lives in `|freqs_cis|`, not in
    the softmax scale.

    Math (numpy), d = x.shape[-1]:
        xc  = x.reshape(*x.shape[:-1], d // 2, 2).view(np.complex64)  # (x0+ix1), ...
        out = (xc * freqs_cis).view(np.float32).reshape(x.shape)

    Args:
        x: Real array [..., seq, rope_dim].
        freqs_cis: Complex rotations [seq, rope_dim // 2].

    Returns:
        Rotated array, same shape and dtype as `x`.
    """
    s, d = x.shape[-2], x.shape[-1]
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], d // 2, 2))
    out = torch.view_as_real(xc * freqs_cis).flatten(-2)
    return out.to(x.dtype)


def apply_rope_interleave(x, cos, sin):
    """Applies GLM (glm4_moe_lite) interleaved RoPE to the last axis.

    Even/odd dims are the interleaved pairs; the output is laid out split-half
    `[rotated-evens | rotated-odds]`. Same rotation as `apply_rope`, different
    output ordering -- self-consistent since q and k share it.

    Math (numpy), c = cos[..., :rope_dim // 2], s = sin[..., :rope_dim // 2]:
        np.concatenate([x[..., 0::2] * c - x[..., 1::2] * s,
                        x[..., 1::2] * c + x[..., 0::2] * s], axis=-1)

    Args:
        x: Real array [..., seq, rope_dim].
        cos: Real array [seq, rope_dim] = concat(freqs, freqs); first half used.
        sin: Real array [seq, rope_dim] = concat(freqs, freqs); first half used.

    Returns:
        Rotated array, same shape and dtype as `x`.
    """
    half = cos.shape[-1] // 2
    c, s_ = cos[..., :half], sin[..., :half]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.cat([x1 * c - x2 * s_, x2 * c + x1 * s_], dim=-1)
    return out.to(x.dtype)


def split_kv_b(kv_b_weight, n_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank):
    """Splits kv_b_proj.weight into per-head up-projections W_UK and W_UV.

    Math (numpy):
        w = kv_b_weight.reshape(n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
        W_UK, W_UV = w[:, :qk_nope_head_dim], w[:, qk_nope_head_dim:]

    Args:
        kv_b_weight: [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank].
        n_heads: Number of attention heads.
        qk_nope_head_dim: Per-head non-RoPE key dim.
        v_head_dim: Per-head value dim.
        kv_lora_rank: Compressed KV latent dim.

    Returns:
        (W_UK, W_UV):
            W_UK: [n_heads, qk_nope_head_dim, kv_lora_rank]  (c_KV -> k_nope)
            W_UV: [n_heads, v_head_dim, kv_lora_rank]        (c_KV -> value)
    """
    w = kv_b_weight.view(n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    W_UK = w[:, :qk_nope_head_dim, :].contiguous()
    W_UV = w[:, qk_nope_head_dim:, :].contiguous()
    return W_UK, W_UV


def mla_unabsorbed(q, c_kv, k_pe, W_UK, W_UV, softmax_scale, causal=True):
    """Computes MLA attention by decompressing the latent to full K/V (prefill).

    Math (numpy), scale = softmax_scale, H = n_heads:
        k_nope = c_kv @ W_UK.transpose(0, 2, 1)             # [H, kv_len, qk_nope]
        value  = c_kv @ W_UV.transpose(0, 2, 1)             # [H, kv_len, v_head_dim]
        k      = np.concatenate([k_nope, broadcast(k_pe)], axis=-1)
        scores = (q @ k.transpose(0, 2, 1)) * scale         # [H, q_len, kv_len]
        attn   = softmax(scores, axis=-1)
        out    = attn @ value                               # [H, q_len, v_head_dim]
        attn_out = out.transpose(1, 0, 2).reshape(q_len, H * v_head_dim)

    Args:
        q: [n_heads, q_len, q_head_dim], q_nope || q_pe (RoPE already on q_pe).
        c_kv: [kv_len, kv_lora_rank], post kv_a_layernorm.
        k_pe: [kv_len, qk_rope_head_dim], RoPE applied, shared across heads.
        W_UK: [n_heads, qk_nope_head_dim, kv_lora_rank].
        W_UV: [n_heads, v_head_dim, kv_lora_rank].
        softmax_scale: Score scale (q_head_dim ** -0.5).
        causal: Whether to apply the causal mask.

    Returns:
        (attn_out, intermediates):
            attn_out: [q_len, n_heads * v_head_dim].
            intermediates: dict with k_nope, value, scores, attn.
    """
    n_heads, q_len, q_head_dim = q.shape
    qk_nope = W_UK.shape[1]
    qk_rope = k_pe.shape[1]
    v_head_dim = W_UV.shape[1]
    kv_len = c_kv.shape[0]

    k_nope = torch.einsum("lr,hdr->hld", c_kv, W_UK)   # [n_heads, kv_len, qk_nope]
    value = torch.einsum("lr,hdr->hld", c_kv, W_UV)    # [n_heads, kv_len, v_head_dim]
    k_pe_b = k_pe.unsqueeze(0).expand(n_heads, kv_len, qk_rope)
    k_full = torch.cat([k_nope, k_pe_b], dim=-1)       # [n_heads, kv_len, q_head_dim]

    scores = torch.einsum("hqd,hkd->hqk", q, k_full) * softmax_scale
    scores = _causal_mask(scores, q_len, kv_len, causal)
    attn = torch.softmax(scores.to(torch.float32), dim=-1)
    out = torch.einsum("hqk,hkd->hqd", attn, value)    # [n_heads, q_len, v_head_dim]
    attn_out = out.transpose(0, 1).reshape(q_len, n_heads * v_head_dim)
    return attn_out, {"k_nope": k_nope, "value": value, "scores": scores, "attn": attn}


def mla_absorbed(q, c_kv, k_pe, W_UK, W_UV, softmax_scale, causal=True):
    """Computes MLA attention directly over the compressed latent (decode).

    Equivalent to `mla_unabsorbed` in exact arithmetic: W_UK is folded into the
    query and W_UV into the output, so attention runs in latent space.

    Math (numpy), q_nope = q[..., :qk_nope], q_pe = q[..., qk_nope:]:
        q_absorbed = q_nope @ W_UK                                 # [H, q_len, kv_lora]
        scores = (q_absorbed @ c_kv.T + q_pe @ k_pe.T) * scale     # [H, q_len, kv_len]
        attn   = softmax(scores, axis=-1)
        ctx    = attn @ c_kv                                       # [H, q_len, kv_lora]
        out    = ctx @ W_UV.transpose(0, 2, 1)                     # [H, q_len, v_head_dim]
        attn_out = out.transpose(1, 0, 2).reshape(q_len, H * v_head_dim)

    Args:
        q: [n_heads, q_len, q_head_dim], q_nope || q_pe (RoPE already on q_pe).
        c_kv: [kv_len, kv_lora_rank], post kv_a_layernorm.
        k_pe: [kv_len, qk_rope_head_dim], RoPE applied, shared across heads.
        W_UK: [n_heads, qk_nope_head_dim, kv_lora_rank].
        W_UV: [n_heads, v_head_dim, kv_lora_rank].
        softmax_scale: Score scale (q_head_dim ** -0.5).
        causal: Whether to apply the causal mask.

    Returns:
        (attn_out, intermediates):
            attn_out: [q_len, n_heads * v_head_dim].
            intermediates: dict with q_absorbed, ctx_latent, scores, attn.
    """
    n_heads, q_len, q_head_dim = q.shape
    qk_nope = W_UK.shape[1]
    qk_rope = k_pe.shape[1]
    kv_len = c_kv.shape[0]
    q_nope = q[..., :qk_nope]            # [n_heads, q_len, qk_nope]
    q_pe = q[..., qk_nope:]              # [n_heads, q_len, qk_rope]

    q_absorbed = torch.einsum("hqd,hdr->hqr", q_nope, W_UK)   # [n_heads, q_len, kv_lora]

    score_nope = torch.einsum("hqr,kr->hqk", q_absorbed, c_kv)
    score_pe = torch.einsum("hqd,kd->hqk", q_pe, k_pe)
    scores = (score_nope + score_pe) * softmax_scale
    scores = _causal_mask(scores, q_len, kv_len, causal)
    attn = torch.softmax(scores.to(torch.float32), dim=-1)

    ctx = torch.einsum("hqk,kr->hqr", attn, c_kv)            # [n_heads, q_len, kv_lora]
    out = torch.einsum("hqr,hdr->hqd", ctx, W_UV)            # [n_heads, q_len, v_head_dim]
    attn_out = out.transpose(0, 1).reshape(q_len, n_heads * W_UV.shape[1])
    return attn_out, {"q_absorbed": q_absorbed, "ctx_latent": ctx,
                      "scores": scores, "attn": attn}


def _causal_mask(scores, q_len, kv_len, causal):
    """Masks scores so query i attends only to keys at or before its position.

    Queries are right-aligned to the keys (offset = kv_len - q_len), supporting a
    q_len < kv_len decode step against a longer cache.

    Args:
        scores: [n_heads, q_len, kv_len].
        q_len: Number of query positions.
        kv_len: Number of key positions.
        causal: If False, returns `scores` unchanged.

    Returns:
        `scores` with masked entries set to -inf.
    """
    if not causal:
        return scores
    offset = kv_len - q_len
    i = torch.arange(q_len).unsqueeze(1)
    j = torch.arange(kv_len).unsqueeze(0)
    mask = (j > (i + offset))
    return scores.masked_fill(mask.unsqueeze(0), float("-inf"))
