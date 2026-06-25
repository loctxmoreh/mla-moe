"""Generate ground-truth ("oracle") activations for a single forward pass of a
DeepSeek-V2-family MLA+MoE model, to validate the C engine's forward()/decode.

Reference: transformers' NATIVE model (transformers 5.x), run in fp32 — mirrors
the C engine's "bf16 weights, fp32 compute" contract.

Supported models (separate dumps/<model>/ each):
  dsv2lite  DeepSeek-V2-Lite   — q_proj (no q-LoRA), complex RoPE, softmax-greedy router
  glm47     GLM-4.7-Flash      — q-LoRA (768), interleaved RoPE, noaux_tc sigmoid router

Two phases, mirroring the C engine's two code paths:
  PREFILL (q_len=N): native attention == UNABSORBED MLA. Dumps logits, per-layer
    attn/mlp outputs, hidden states, layer-0 attn internals, first-MoE-layer router.
  DECODE  (q_len=1): one step with KV cache. HF logits are the decode ground truth.
    layer-0 attention is recomputed with the ABSORBED MLA reference and asserted
    equal to BOTH the UNABSORBED reference and HF (≤ tol). Dumps absorbed internals.

Run:  cd tests/oracle && ../../.venv/bin/python gen_oracle.py [dsv2lite|glm47]
"""

import os
import sys
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mla_reference as ref

MODELS = {
    "dsv2lite": dict(
        dir="/remote/vast0/share-mv/deepseek-ai/DeepSeek-V2-Lite",
        prompt="The capital of France is", rope="complex", router="softmax_greedy"),
    "glm47": dict(
        dir="/remote/vast0/share-mv/zai-org/GLM-4.7-Flash",
        prompt="The capital of France is", rope="interleave", router="noaux_tc"),
}
TOL = 2e-3

torch.manual_seed(0)
torch.set_grad_enabled(False)

_manifest = {}
captured = {}
PHASE = "prefill"


def dump(out_dir, name, tensor):
    arr = tensor.detach().to(torch.float32).cpu().numpy().astype("<f4")
    arr.tofile(os.path.join(out_dir, name + ".f32.bin"))
    _manifest[name] = {"file": name + ".f32.bin", "shape": list(arr.shape),
                       "dtype": "float32", "order": "C"}


def dump_int(out_dir, name, data):
    arr = np.asarray(data).astype("<i4")
    arr.tofile(os.path.join(out_dir, name + ".i32.bin"))
    _manifest[name] = {"file": name + ".i32.bin", "shape": list(arr.shape),
                       "dtype": "int32", "order": "C"}


def make_io_hooks(model, n_layers):
    def pre(idx, tag):
        def h(mod, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            captured[(idx, tag, PHASE)] = hs.detach()[0].float()
        return h

    def attn_post(idx):
        def h(mod, args, output):
            out = output[0] if isinstance(output, tuple) else output
            captured[(idx, "attn_out", PHASE)] = out.detach()[0].float()
        return h

    def mlp_post(idx):
        def h(mod, args, output):
            captured[(idx, "mlp_out", PHASE)] = output.detach()[0].float()
        return h

    for i in range(n_layers):
        lyr = model.model.layers[i]
        lyr.self_attn.register_forward_pre_hook(pre(i, "attn_in"), with_kwargs=True)
        lyr.self_attn.register_forward_hook(attn_post(i))
        lyr.mlp.register_forward_pre_hook(pre(i, "mlp_in"), with_kwargs=True)
        lyr.mlp.register_forward_hook(mlp_post(i))


def make_rope(model, kind):
    """Return rope_apply(x, positions) matching the model, x = [..., seq, rope_dim]."""
    rot = model.model.rotary_emb

    def apply(x, positions):
        pos = torch.tensor(positions, dtype=torch.long).unsqueeze(0)
        dummy = torch.zeros(1, 1, dtype=torch.float32)
        if kind == "complex":
            fc = rot(dummy, pos)[0]            # [seq, rope_dim//2] complex
            return ref.apply_rope(x, fc)
        cos, sin = rot(dummy, pos)             # each [1, seq, rope_dim]
        return ref.apply_rope_interleave(x, cos[0], sin[0])
    return apply


def build_q_kv(sa, normed_hidden, cfg):
    """self_attn input (input_layernorm'd) -> q (no rope), latent c_kv, raw k_pe."""
    n_heads, qk_nope, qk_rope = cfg["n_heads"], cfg["qk_nope"], cfg["qk_rope"]
    q_head_dim = qk_nope + qk_rope
    seq = normed_hidden.shape[0]
    if cfg["q_lora"]:
        qa = ref.rms_norm(normed_hidden @ sa.q_a_proj.weight.float().T,
                          sa.q_a_layernorm.weight.float(), sa.q_a_layernorm.variance_epsilon)
        q = qa @ sa.q_b_proj.weight.float().T
    else:
        q = normed_hidden @ sa.q_proj.weight.float().T
    q = q.view(seq, n_heads, q_head_dim).transpose(0, 1)
    comp = normed_hidden @ sa.kv_a_proj_with_mqa.weight.float().T
    c_kv = ref.rms_norm(comp[:, :cfg["kv_lora"]], sa.kv_a_layernorm.weight.float(),
                        sa.kv_a_layernorm.variance_epsilon)
    return q, c_kv, comp[:, cfg["kv_lora"]:]


def route(kind, moe_in, gate, cfg):
    """Recompute the first-MoE-layer router. Returns (scores, topk_idx, topk_w)."""
    logits = moe_in.float() @ gate.weight.float().T          # [tokens, n_experts]
    k, rs = cfg["top_k"], cfg["routed_scaling"]
    if kind == "softmax_greedy":                              # DeepSeek-V2
        scores = torch.softmax(logits, dim=-1)
        w, idx = torch.topk(scores, k, dim=-1, sorted=True)
        if cfg["norm_topk"]:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        return scores, idx, w * rs
    # noaux_tc (GLM): sigmoid scores, bias-corrected choice, gather raw scores, norm
    scores = logits.sigmoid()
    bias = gate.e_score_correction_bias.float()
    idx = torch.topk(scores + bias, k, dim=-1, sorted=True)[1]
    w = scores.gather(1, idx)
    if cfg["norm_topk"]:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return scores, idx, w * rs


def main(model_key):
    spec = MODELS[model_key]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps", model_key)
    os.makedirs(out_dir, exist_ok=True)
    _manifest.clear(); captured.clear()
    print(f"[{model_key}] loading {spec['dir']} (fp32, eager) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(spec["dir"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["dir"], torch_dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True)
    model.eval()
    c = model.config
    cfg = dict(n_heads=c.num_attention_heads, qk_nope=c.qk_nope_head_dim,
               qk_rope=c.qk_rope_head_dim, kv_lora=c.kv_lora_rank, v_head_dim=c.v_head_dim,
               hidden=c.hidden_size, n_layers=c.num_hidden_layers,
               n_routed=c.n_routed_experts, top_k=c.num_experts_per_tok,
               q_lora=getattr(c, "q_lora_rank", None) is not None,
               routed_scaling=float(getattr(c, "routed_scaling_factor", 1.0)),
               norm_topk=bool(getattr(c, "norm_topk_prob", False)),
               first_k_dense=getattr(c, "first_k_dense_replace", 1))
    sa0 = model.model.layers[0].self_attn
    scaling = float(sa0.scaling)
    qk_nope = cfg["qk_nope"]
    moe_layer = cfg["first_k_dense"]                         # first MoE layer
    print(f"  layers={cfg['n_layers']} heads={cfg['n_heads']} "
          f"q_head_dim={qk_nope+cfg['qk_rope']} v_head={cfg['v_head_dim']} "
          f"q_lora={cfg['q_lora']} scaling={scaling:.6f} rope={spec['rope']} "
          f"router={spec['router']} moe_layer={moe_layer}", flush=True)

    make_io_hooks(model, cfg["n_layers"])
    rope = make_rope(model, spec["rope"])
    o_proj = sa0.o_proj.weight.float()
    W_UK, W_UV = ref.split_kv_b(sa0.kv_b_proj.weight.float(), cfg["n_heads"],
                                qk_nope, cfg["v_head_dim"], cfg["kv_lora"])

    ids = tok(spec["prompt"], return_tensors="pt").input_ids
    seq = ids.shape[1]
    print(f"  prompt={spec['prompt']!r} ids={ids[0].tolist()} seq={seq}", flush=True)
    dump_int(out_dir, "input_ids", ids[0].tolist())

    # ----------------------------- PREFILL (unabsorbed) -----------------------------
    global PHASE
    PHASE = "prefill"
    out = model(ids, use_cache=True, output_hidden_states=True)
    past = out.past_key_values
    logits_pf = out.logits[0].float()
    dump(out_dir, "prefill_logits", logits_pf)
    dump(out_dir, "prefill_hidden_states",
         torch.stack([h[0].float() for h in out.hidden_states]))
    for i in range(cfg["n_layers"]):
        dump(out_dir, f"prefill_layer{i:02d}_attn_out", captured[(i, "attn_out", "prefill")])
        dump(out_dir, f"prefill_layer{i:02d}_mlp_out", captured[(i, "mlp_out", "prefill")])

    scores, ti, tw = route(spec["router"], captured[(moe_layer, "mlp_in", "prefill")],
                           model.model.layers[moe_layer].mlp.gate, cfg)
    dump(out_dir, f"prefill_moe{moe_layer}_router_scores", scores)
    dump_int(out_dir, f"prefill_moe{moe_layer}_topk_idx", ti.cpu().numpy())
    dump(out_dir, f"prefill_moe{moe_layer}_topk_w", tw)

    # layer-0 attention internals + UNABSORBED self-check vs HF
    q, c_kv, k_pe_raw = build_q_kv(sa0, captured[(0, "attn_in", "prefill")], cfg)
    q_pe = rope(q[..., qk_nope:], list(range(seq)))
    k_pe = rope(k_pe_raw, list(range(seq)))
    q_roped = torch.cat([q[..., :qk_nope], q_pe], dim=-1)
    attn_un, mid = ref.mla_unabsorbed(q_roped, c_kv, k_pe, W_UK, W_UV, scaling)
    err = (attn_un @ o_proj.T - captured[(0, "attn_out", "prefill")]).abs().max().item()
    print(f"  [prefill] layer0 unabsorbed attn_out vs HF: max_abs_err={err:.2e}", flush=True)
    assert err < TOL, f"prefill unabsorbed mismatch {err}"
    for nm, t in [("q_nope", q[..., :qk_nope]), ("q_pe", q_pe), ("c_kv", c_kv),
                  ("k_pe", k_pe), ("k_nope", mid["k_nope"]), ("value", mid["value"]),
                  ("attn_scores", mid["scores"]), ("attn_weights", mid["attn"])]:
        dump(out_dir, f"prefill_layer0_{nm}", t)
    dump(out_dir, "layer0_W_UK", W_UK)
    dump(out_dir, "layer0_W_UV", W_UV)

    # ----------------------------- DECODE (absorbed) -----------------------------
    PHASE = "decode"
    next_id = int(logits_pf[-1].argmax())
    print(f"  [decode] next_id={next_id} ({tok.decode([next_id])!r})", flush=True)
    dump_int(out_dir, "decode_input_id", [next_id])
    out2 = model(torch.tensor([[next_id]]), past_key_values=past,
                 position_ids=torch.tensor([[seq]]), use_cache=True)
    dump(out_dir, "decode_logits", out2.logits[0].float())

    kv_len = seq + 1
    nh_all = torch.cat([captured[(0, "attn_in", "prefill")],
                        captured[(0, "attn_in", "decode")]], dim=0)
    _, c_kv_all, k_pe_raw_all = build_q_kv(sa0, nh_all, cfg)
    k_pe_all = rope(k_pe_raw_all, list(range(kv_len)))
    q_d, _, _ = build_q_kv(sa0, captured[(0, "attn_in", "decode")], cfg)
    q_d_pe = rope(q_d[..., qk_nope:], [seq])
    q_d_roped = torch.cat([q_d[..., :qk_nope], q_d_pe], dim=-1)

    attn_abs, mab = ref.mla_absorbed(q_d_roped, c_kv_all, k_pe_all, W_UK, W_UV,
                                     scaling, causal=False)
    attn_un2, _ = ref.mla_unabsorbed(q_d_roped, c_kv_all, k_pe_all, W_UK, W_UV,
                                     scaling, causal=False)
    e_ab_un = (attn_abs - attn_un2).abs().max().item()
    e_ab_hf = (attn_abs @ o_proj.T - captured[(0, "attn_out", "decode")]).abs().max().item()
    print(f"  [decode] layer0 absorbed vs unabsorbed: max_abs_err={e_ab_un:.2e}", flush=True)
    print(f"  [decode] layer0 absorbed attn_out vs HF:  max_abs_err={e_ab_hf:.2e}", flush=True)
    assert e_ab_un < TOL and e_ab_hf < TOL, f"absorbed mismatch un={e_ab_un} hf={e_ab_hf}"
    dump(out_dir, "decode_layer0_c_kv_cache", c_kv_all)
    dump(out_dir, "decode_layer0_k_pe_cache", k_pe_all)
    dump(out_dir, "decode_layer0_q_absorbed", mab["q_absorbed"])
    dump(out_dir, "decode_layer0_ctx_latent", mab["ctx_latent"])
    dump(out_dir, "decode_layer0_attn_weights", mab["attn"])
    dump(out_dir, "decode_layer0_attn_out", attn_abs @ o_proj.T)

    meta = {"model": model_key, "model_dir": spec["dir"], "prompt": spec["prompt"],
            "seq_len": seq, "decode_pos": seq, "next_id": next_id, "scaling": scaling,
            "rope": spec["rope"], "router": spec["router"], "moe_layer": moe_layer,
            "config": cfg, "tol": TOL, "tensors": _manifest,
            "notes": {
                "kv_b_split": "head-INTERLEAVED [k_nope|value] per head; layer0_W_UK/W_UV "
                              "are the correct split (per-head stride (qk_nope+v_head)*kv_lora).",
                "rope": f"{spec['rope']} on the qk_rope slice; softmax uses plain "
                        f"scaling=q_head_dim**-0.5.",
                "cache": "decode caches c_kv(kv_lora) || k_pe(qk_rope) per pos."}}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {len(_manifest)} tensors + manifest.json to {out_dir}")
    print(f"OK [{model_key}] — prefill (unabsorbed) + decode (absorbed) generated & cross-checked.\n")


if __name__ == "__main__":
    keys = sys.argv[1:] or ["dsv2lite"]
    for k in keys:
        main(k)
