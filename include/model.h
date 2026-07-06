/* Model types + loader. The Config/ModelWeights/RunState/Transformer structs and
 * the config.json + safetensors loading live here so run.c holds only the math
 * (forward passes, kernels). Implemented in src/model_load.c. */
#ifndef MLA_MODEL_H
#define MLA_MODEL_H

#include "tensor.h"              /* bf16_t */
#include "safetensors_loader.h" /* TensorStore, Tensor, st_* */
#include "tokenizer.h"          /* Tokenizer, tokenizer_* */

/* Mirrors llama2.c's Config; populated from config.json at load time. */
typedef struct {
    int   n_layers;
    int   hidden_size;
    int   n_heads;
    int   n_kv_heads;         /* MLA: == n_heads (no GQA) */
    int   q_lora_rank;        /* 0 => single q_proj (DSV2); >0 => q_a/q_b LoRA (GLM) */
    int   first_k_dense;      /* layers [0, first_k_dense) are dense FFN, not MoE */
    int   kv_lora_rank;       /* d_c: compressed KV latent dim */
    int   qk_rope_head_dim;   /* per-head RoPE-bearing q/k slice */
    int   qk_nope_head_dim;   /* per-head non-RoPE q/k slice */
    int   v_head_dim;
    int   n_routed_experts;
    int   n_shared_experts;
    int   n_experts_per_tok;  /* top-k routed experts per token */
    int   dense_inter_size;   /* mlp.{gate,up,down}_proj width on dense layers */
    int   moe_inter_size;     /* per-expert FFN width on MoE layers */
    int   vocab_size;
    int   max_seq_len;
    float rms_eps;            /* decoder norms: input/post_attn/final */
    float mla_norm_eps;       /* q_a_layernorm & kv_a_layernorm (GLM uses 1e-6 here,
                               * not the config 1e-5 — q_a's tiny variance is eps-sensitive) */
    float softmax_scale;      /* q_head_dim**-0.5 (no mscale for either model) */
    /* YaRN RoPE params (for inv_freq construction at build time; factor==1 =>
     * plain RoPE, which is how GLM uses it). */
    float rope_theta, rope_factor, rope_beta_fast, rope_beta_slow, rope_orig_max;
    int   rope_interleaved;   /* 0: dsv2 complex adjacent-pair; 1: GLM split-half */
    /* MoE router flavor */
    int   router_sigmoid;     /* 0: softmax (dsv2); 1: sigmoid + e_score bias (GLM) */
    int   norm_topk;          /* normalize the top-k weights before scaling */
    float routed_scaling;     /* multiply routed weights (1.0 dsv2, 1.8 GLM) */
} Config;

/* bf16_t* point into TensorStore mmaps (zero copy); per-layer arrays are
 * malloc'd pointer arrays. kv_b_proj is split into W_UK || W_UV by pointer
 * arithmetic at load time. */
typedef struct {
    /* Token embedding table: [vocab_size, hidden_size] */
    bf16_t *embed_tokens;

    /* Per-layer MLA attention weights.
     * Query has two variants — exactly one is populated per model:
     *   q_lora_rank == 0 (DSV2-Lite): q_proj only; q_a/q_b/q_a_layernorm are NULL.
     *   q_lora_rank  > 0 (GLM):       q_a_proj/q_b_proj/q_a_layernorm; q_proj is NULL. */
    bf16_t **q_proj;         /* [n_layers]: [n_heads*qk_head_dim, hidden_size]       W_Q (no LoRA) */
    bf16_t **q_a_proj;       /* [n_layers]: [q_lora_rank, hidden_size]               W_DQ  */
    bf16_t **q_b_proj;       /* [n_layers]: [n_heads*qk_head_dim, q_lora_rank]       W_UQ  */
    bf16_t **kv_a_proj;      /* [n_layers]: [kv_lora_rank+qk_rope_head_dim, hidden]  W_DKV */
    /* Per-head MLA up-projections, split from kv_b_proj (head-INTERLEAVED, zero-copy).
     * kv_b_proj.weight is [n_heads*(qk_nope_head_dim+v_head_dim), kv_lora_rank]; per
     * head the rows are [W_UK_h (qk_nope) | W_UV_h (v_head)]. W_UK[l]/W_UV[l] point at
     * head 0; both use the SAME per-head stride (qk_nope_head_dim+v_head_dim)*kv_lora_rank:
     * head h -> W_UK[l] + h*stride, W_UV[l] + h*stride. */
    bf16_t **W_UK;           /* [n_layers]: head 0's [qk_nope_head_dim, kv_lora_rank] */
    bf16_t **W_UV;           /* [n_layers]: head 0's [v_head_dim,       kv_lora_rank] */
    bf16_t **o_proj;         /* [n_layers]: [hidden_size, n_heads*v_head_dim]        */

    /* Per-layer RMSNorm weights */
    bf16_t **input_layernorm;      /* [n_layers]: [hidden_size] */
    bf16_t **post_attn_norm;       /* [n_layers]: [hidden_size] */
    bf16_t **q_a_layernorm;        /* [n_layers]: [q_lora_rank] */
    bf16_t **kv_a_layernorm;       /* [n_layers]: [kv_lora_rank] */

    /* Per-layer FFN. For any given layer, EITHER the dense fields below OR the MoE
     * fields (moe_gate, expert_*, shared_*) are populated — never both.
     * Layers [0, first_k_dense) are dense; the rest are MoE. */

    /* Dense FFN weights (mlp.{gate,up,down}_proj) — non-NULL only for dense layers */
    bf16_t **dense_gate;     /* [n_layers]: [dense_inter_size, hidden_size] */
    bf16_t **dense_up;
    bf16_t **dense_down;

    /* Per-layer MoE routing — non-NULL only for MoE layers */
    bf16_t **moe_gate;       /* [n_layers]: [n_routed_experts, hidden_size] */
    float  **moe_gate_bias;  /* [n_layers]: mlp.gate.e_score_correction_bias — GLM only,
                              * stored F32 in the checkpoint (points into mmap); else NULL */

    /* Per-layer routed expert FFN weights: [n_layers][n_experts] (NULL on dense layers) */
    bf16_t ***expert_gate;   /* gate_proj: [inter_size, hidden_size] */
    bf16_t ***expert_up;     /* up_proj  : [inter_size, hidden_size] */
    bf16_t ***expert_down;   /* down_proj: [hidden_size, inter_size] */

    /* Per-layer shared expert FFN weights (NULL on dense layers) */
    bf16_t **shared_gate;    /* [n_layers]: [shared_inter_size, hidden_size] */
    bf16_t **shared_up;
    bf16_t **shared_down;

    /* Final RMSNorm + LM head */
    bf16_t *norm;            /* [hidden_size] */
    bf16_t *lm_head;         /* [vocab_size, hidden_size] */
} ModelWeights;

/* Per-step scratch buffers (owned, float for now). KV cache stores the
 * compressed MLA latent c_KV, not full K/V — the key MLA vs GQA difference. */
typedef struct {
    float *x;           /* current token hidden state: [hidden_size] */
    float *xb;          /* residual branch buffer:     [hidden_size] */
    float *xb2;         /* second residual buffer:     [hidden_size] */

    /* MLA attention intermediates */
    float *q;           /* query:  [n_heads * (qk_nope_head_dim + qk_rope_head_dim)] */
    float *q_a;         /* q-LoRA latent (GLM only): [q_lora_rank]; NULL for dsv2 */
    float *c_kv;        /* compressed KV latent: [kv_lora_rank + qk_rope_head_dim] */
    float *att;         /* attention scores:     [n_heads * max_seq_len] */

    /* KV cache — latent form (the MLA-specific design) */
    float *kv_cache;    /* [n_layers * max_seq_len * (kv_lora_rank + qk_rope_head_dim)] */

    /* MoE FFN intermediates */
    float *moe_logits;  /* router logits: [n_routed_experts] */
    float *expert_out;  /* single expert output: [hidden_size] */
    float *hb;          /* FFN hidden buffer: [max(inter_size, shared_inter_size)] */
    float *hb2;         /* second FFN buffer */

    /* Logits */
    float *logits;      /* [vocab_size] */
} RunState;

/* Top-level container; TensorStore owns the mmaps, ModelWeights points in. */
typedef struct {
    Config       config;
    ModelWeights weights;
    RunState     state;
    TensorStore *store;
    float       *rope_inv_freq;   /* [qk_rope_head_dim/2], YaRN-interpolated */
    Tokenizer   *tokenizer;       /* NULL if model_dir has no tokenizer.json */
} Transformer;

/* Build/tear down a Transformer from <model_dir> (config.json +
 * model.safetensors.index.json + shards). Defined in src/model_load.c. */
void build_transformer(Transformer *t, const char *model_dir);
void free_transformer(Transformer *t);

#endif /* MLA_MODEL_H */
