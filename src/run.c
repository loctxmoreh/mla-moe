/* MLA-MOE inference entry point; structure follows llama2.c.
 * forward_unabsorbed() (whole-prompt prefill) and forward_absorbed() (one-token
 * decode) are both oracle-validated for dsv2lite AND glm47 within tol=2e-3.
 * Config (incl. model family) is read from <model_dir>/config.json. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tensor.h"
#include "safetensors_loader.h"

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
} Transformer;

/* Checked lookup: fail with the name instead of segfaulting on a NULL st_get.
 * Used for tensors that must exist for the selected model; genuinely-optional
 * tensors (e.g. the GLM router bias) use st_get directly and tolerate NULL. */
static bf16_t *must_get(TensorStore *s, const char *name) {
    const Tensor *t = st_get(s, name);
    if (!t) {
        fprintf(stderr, "build_model_weights: tensor not found in checkpoint: %s\n", name);
        exit(1);
    }
    return t->data;
}

/* Resolve tensor names into weight pointers; allocates per-layer arrays,
 * never copies tensor data. */
static void build_model_weights(Transformer *t) {
    Config       *c = &t->config;
    ModelWeights *w = &t->weights;
    TensorStore  *s =  t->store;

    /* --- embed / norm / lm_head (global, one tensor each) --- */
    w->embed_tokens = must_get(s, "model.embed_tokens.weight");
    w->norm         = must_get(s, "model.norm.weight");
    w->lm_head      = must_get(s, "lm_head.weight");

    /* --- allocate per-layer pointer arrays --- */
    w->q_proj        = malloc(c->n_layers * sizeof(bf16_t*));
    w->q_a_proj      = malloc(c->n_layers * sizeof(bf16_t*));
    w->q_b_proj      = malloc(c->n_layers * sizeof(bf16_t*));
    w->kv_a_proj     = malloc(c->n_layers * sizeof(bf16_t*));
    w->W_UK          = malloc(c->n_layers * sizeof(bf16_t*));
    w->W_UV          = malloc(c->n_layers * sizeof(bf16_t*));
    w->o_proj        = malloc(c->n_layers * sizeof(bf16_t*));
    w->input_layernorm = malloc(c->n_layers * sizeof(bf16_t*));
    w->post_attn_norm  = malloc(c->n_layers * sizeof(bf16_t*));
    w->q_a_layernorm   = malloc(c->n_layers * sizeof(bf16_t*));
    w->kv_a_layernorm  = malloc(c->n_layers * sizeof(bf16_t*));
    w->dense_gate      = malloc(c->n_layers * sizeof(bf16_t*));
    w->dense_up        = malloc(c->n_layers * sizeof(bf16_t*));
    w->dense_down      = malloc(c->n_layers * sizeof(bf16_t*));
    w->moe_gate        = malloc(c->n_layers * sizeof(bf16_t*));
    w->moe_gate_bias   = malloc(c->n_layers * sizeof(float*));
    w->shared_gate     = malloc(c->n_layers * sizeof(bf16_t*));
    w->shared_up       = malloc(c->n_layers * sizeof(bf16_t*));
    w->shared_down     = malloc(c->n_layers * sizeof(bf16_t*));
    w->expert_gate = malloc(c->n_layers * sizeof(bf16_t**));
    w->expert_up   = malloc(c->n_layers * sizeof(bf16_t**));
    w->expert_down = malloc(c->n_layers * sizeof(bf16_t**));

    char name[256];
    for (int l = 0; l < c->n_layers; l++) {
        /* attention — query has two variants (see ModelWeights comment) */
        if (c->q_lora_rank > 0) {
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_a_proj.weight", l);
            w->q_a_proj[l] = must_get(s, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_b_proj.weight", l);
            w->q_b_proj[l] = must_get(s, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_a_layernorm.weight", l);
            w->q_a_layernorm[l] = must_get(s, name);
            w->q_proj[l] = NULL;
        } else {
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", l);
            w->q_proj[l] = must_get(s, name);
            w->q_a_proj[l] = w->q_b_proj[l] = w->q_a_layernorm[l] = NULL;
        }

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.kv_a_proj_with_mqa.weight", l);
        w->kv_a_proj[l] = must_get(s, name);

        /* Split kv_b_proj into W_UK/W_UV at head 0 — no copy. Layout is head-
         * INTERLEAVED ([W_UK_h | W_UV_h] per head), so W_UV starts after head 0's
         * W_UK, NOT after all heads' W_UK. Per-head stride is applied at use. */
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.kv_b_proj.weight", l);
        bf16_t *kv_b = must_get(s, name);
        w->W_UK[l] = kv_b;
        w->W_UV[l] = kv_b + (size_t)c->qk_nope_head_dim * c->kv_lora_rank;

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", l);
        w->o_proj[l] = must_get(s, name);

        /* norms */
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
        w->input_layernorm[l] = must_get(s, name);

        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", l);
        w->post_attn_norm[l] = must_get(s, name);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.kv_a_layernorm.weight", l);
        w->kv_a_layernorm[l] = must_get(s, name);

        /* FFN — dense for the first first_k_dense layers, MoE thereafter */
        if (l < c->first_k_dense) {
            snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", l);
            w->dense_gate[l] = must_get(s, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", l);
            w->dense_up[l]   = must_get(s, name);
            snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", l);
            w->dense_down[l] = must_get(s, name);

            w->moe_gate[l] = NULL; w->moe_gate_bias[l] = NULL;
            w->shared_gate[l] = w->shared_up[l] = w->shared_down[l] = NULL;
            w->expert_gate[l] = w->expert_up[l] = w->expert_down[l] = NULL;
            continue;
        }
        w->dense_gate[l] = w->dense_up[l] = w->dense_down[l] = NULL;

        /* MoE routing */
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", l);
        w->moe_gate[l] = must_get(s, name);

        /* optional router bias (GLM has it, DSV2 does not) */
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.e_score_correction_bias", l);
        const Tensor *bias = st_get(s, name);   /* F32 in checkpoint, zero-copy */
        w->moe_gate_bias[l] = bias ? (float *)bias->data : NULL;

        /* shared expert */
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_experts.gate_proj.weight", l);
        w->shared_gate[l] = must_get(s, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_experts.up_proj.weight", l);
        w->shared_up[l]   = must_get(s, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_experts.down_proj.weight", l);
        w->shared_down[l] = must_get(s, name);

        /* routed experts */
        w->expert_gate[l] = malloc(c->n_routed_experts * sizeof(bf16_t*));
        w->expert_up[l]   = malloc(c->n_routed_experts * sizeof(bf16_t*));
        w->expert_down[l] = malloc(c->n_routed_experts * sizeof(bf16_t*));
        for (int e = 0; e < c->n_routed_experts; e++) {
            snprintf(name, sizeof(name),
                "model.layers.%d.mlp.experts.%d.gate_proj.weight", l, e);
            w->expert_gate[l][e] = must_get(s, name);
            snprintf(name, sizeof(name),
                "model.layers.%d.mlp.experts.%d.up_proj.weight", l, e);
            w->expert_up[l][e]   = must_get(s, name);
            snprintf(name, sizeof(name),
                "model.layers.%d.mlp.experts.%d.down_proj.weight", l, e);
            w->expert_down[l][e] = must_get(s, name);
        }
    }
}

static void malloc_run_state(RunState *s, const Config *c) {
    int h = c->hidden_size;
    int kv_dim   = c->kv_lora_rank + c->qk_rope_head_dim;
    int q_dim    = c->n_heads * (c->qk_nope_head_dim + c->qk_rope_head_dim);
    /* widest FFN: dense, the (n_shared * moe) shared-expert MLP, or one routed expert */
    int shared   = c->moe_inter_size * c->n_shared_experts;
    int inter    = c->dense_inter_size;
    if (shared > inter)         inter = shared;
    if (c->moe_inter_size > inter) inter = c->moe_inter_size;

    s->x          = calloc(h, sizeof(float));
    s->xb         = calloc(h, sizeof(float));
    s->xb2        = calloc(h, sizeof(float));
    s->q          = calloc(q_dim, sizeof(float));
    s->q_a        = c->q_lora_rank > 0 ? calloc(c->q_lora_rank, sizeof(float)) : NULL;
    s->c_kv       = calloc(kv_dim, sizeof(float));
    s->att        = calloc((size_t)c->n_heads * c->max_seq_len, sizeof(float));
    s->kv_cache   = calloc((size_t)c->n_layers * c->max_seq_len * kv_dim, sizeof(float));
    s->moe_logits = calloc(c->n_routed_experts, sizeof(float));
    s->expert_out = calloc(h, sizeof(float));
    s->hb         = calloc(inter, sizeof(float));
    s->hb2        = calloc(inter, sizeof(float));
    s->logits     = calloc(c->vocab_size, sizeof(float));
}

static void free_run_state(RunState *s) {
    free(s->x);   free(s->xb);  free(s->xb2);
    free(s->q);   free(s->q_a); free(s->c_kv); free(s->att);
    free(s->kv_cache);
    free(s->moe_logits); free(s->expert_out);
    free(s->hb);  free(s->hb2); free(s->logits);
}

static void free_model_weights(ModelWeights *w, int n_layers) {
    /* free pointer arrays only; bf16 data is owned by TensorStore.
     * Inner expert arrays exist only for MoE layers (NULL on dense layers). */
    for (int l = 0; l < n_layers; l++) {
        free(w->expert_gate[l]);
        free(w->expert_up[l]);
        free(w->expert_down[l]);
    }
    free(w->q_proj);      free(w->q_a_proj);  free(w->q_b_proj);  free(w->kv_a_proj);
    free(w->W_UK);        free(w->W_UV);      free(w->o_proj);
    free(w->input_layernorm); free(w->post_attn_norm);
    free(w->q_a_layernorm);   free(w->kv_a_layernorm);
    free(w->dense_gate);  free(w->dense_up);  free(w->dense_down);
    free(w->moe_gate);    free(w->moe_gate_bias);
    free(w->shared_gate); free(w->shared_up); free(w->shared_down);
    free(w->expert_gate); free(w->expert_up); free(w->expert_down);
}

/* ---- minimal config.json reader -----------------------------------------
 * config.json is flat JSON (one nested object, rope_scaling). We only need
 * scalar lookups by key, so a substring scan suffices — no real parser. Each
 * search key includes its quotes, so e.g. "factor" never matches inside
 * "routed_scaling_factor" and "max_position_embeddings" never matches inside
 * "original_max_position_embeddings". */
static char *read_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "read %s\n", path); exit(1); }
    buf[n] = '\0'; fclose(f);
    return buf;
}

/* Pointer to the first char of key's value (past `"key":` and whitespace), or NULL. */
static const char *json_value(const char *buf, const char *key) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(buf, pat);
    if (!p) return NULL;
    p += strlen(pat);
    while (*p && *p != ':') p++;
    if (*p == ':') p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}
static long json_int(const char *buf, const char *key, long dflt) {
    const char *p = json_value(buf, key);
    return (!p || *p == 'n') ? dflt : strtol(p, NULL, 10);   /* 'n' => null */
}
static double json_double(const char *buf, const char *key, double dflt) {
    const char *p = json_value(buf, key);
    return (!p || *p == 'n') ? dflt : strtod(p, NULL);
}
static int json_bool(const char *buf, const char *key, int dflt) {
    const char *p = json_value(buf, key);
    return p ? (*p == 't') : dflt;
}
static int json_str_is(const char *buf, const char *key, const char *val) {
    const char *p = json_value(buf, key);
    if (!p || *p != '"') return 0;
    p++;
    size_t n = strlen(val);
    return strncmp(p, val, n) == 0 && p[n] == '"';
}

/* Cap on the latent KV cache length (config max_position_embeddings is huge:
 * 163840 / 202752). Prompt + generated tokens must fit under this. */
#define KV_CACHE_CAP 4096

/* Build Config from <model_dir>/config.json. Model-family behavior is derived,
 * not hardcoded: rope layout from model_type, router flavor from topk_method,
 * YaRN params from rope_scaling (absent => plain rope). */
static Config config_from_json(const char *model_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    char *b = read_text_file(path);

    Config c = {0};
    c.n_layers          = (int)json_int(b, "num_hidden_layers", 0);
    c.hidden_size       = (int)json_int(b, "hidden_size", 0);
    c.n_heads           = (int)json_int(b, "num_attention_heads", 0);
    c.n_kv_heads        = c.n_heads;                       /* MLA: no GQA */
    c.q_lora_rank       = (int)json_int(b, "q_lora_rank", 0);   /* null => 0 (no q-LoRA) */
    c.first_k_dense     = (int)json_int(b, "first_k_dense_replace", 0);
    c.kv_lora_rank      = (int)json_int(b, "kv_lora_rank", 0);
    c.qk_rope_head_dim  = (int)json_int(b, "qk_rope_head_dim", 0);
    c.qk_nope_head_dim  = (int)json_int(b, "qk_nope_head_dim", 0);
    c.v_head_dim        = (int)json_int(b, "v_head_dim", 0);
    c.n_routed_experts  = (int)json_int(b, "n_routed_experts", 0);
    c.n_shared_experts  = (int)json_int(b, "n_shared_experts", 0);
    c.n_experts_per_tok = (int)json_int(b, "num_experts_per_tok", 0);
    c.dense_inter_size  = (int)json_int(b, "intermediate_size", 0);
    c.moe_inter_size    = (int)json_int(b, "moe_intermediate_size", 0);
    c.vocab_size        = (int)json_int(b, "vocab_size", 0);
    long maxpos         = json_int(b, "max_position_embeddings", KV_CACHE_CAP);
    c.max_seq_len       = maxpos < KV_CACHE_CAP ? (int)maxpos : KV_CACHE_CAP;
    c.rms_eps           = (float)json_double(b, "rms_norm_eps", 1e-5);
    c.mla_norm_eps      = 1e-6f;   /* q_a/kv_a RMSNorm class default — not in config.json */
    c.softmax_scale     = 1.0f / sqrtf((float)(c.qk_nope_head_dim + c.qk_rope_head_dim));
    c.rope_theta        = (float)json_double(b, "rope_theta", 10000.0);
    /* rope_scaling (YaRN): present => factor from it; absent/null => plain (factor 1).
     * "factor"/"beta_*"/"original_max_position_embeddings" only occur inside it. */
    c.rope_factor       = (float)json_double(b, "factor", 1.0);
    c.rope_beta_fast    = (float)json_double(b, "beta_fast", 32.0);
    c.rope_beta_slow    = (float)json_double(b, "beta_slow", 1.0);
    c.rope_orig_max     = (float)json_double(b, "original_max_position_embeddings", 4096.0);
    c.rope_interleaved  = json_str_is(b, "model_type", "glm4_moe_lite");  /* else complex */
    c.router_sigmoid    = json_str_is(b, "topk_method", "noaux_tc");      /* else softmax-greedy */
    c.norm_topk         = json_bool(b, "norm_topk_prob", 0);
    c.routed_scaling    = (float)json_double(b, "routed_scaling_factor", 1.0);
    free(b);

    if (c.n_layers <= 0 || c.hidden_size <= 0 || c.kv_lora_rank <= 0) {
        fprintf(stderr, "config_from_json: missing/invalid fields in %s\n", path);
        exit(1);
    }
    return c;
}

/* DeepSeek-V2 YaRN inv_freq: blends interpolated/extrapolated freqs over a
 * correction range. Mirrors transformers' DeepseekV2YarnRotaryEmbedding init.
 * (mscale == mscale_all_dim for dsv2lite, so the freq magnitude is 1 — only the
 * frequencies are interpolated, not the rotation amplitude.) */
static float *build_rope_inv_freq(const Config *c) {
    int dim  = c->qk_rope_head_dim;   /* 64 */
    int half = dim / 2;               /* 32 */
    float base = c->rope_theta, factor = c->rope_factor;
    /* correction range: dims rotating faster than beta_fast use extrapolation,
     * slower than beta_slow use interpolation, linear ramp in between */
    float lo_d = (dim * logf(c->rope_orig_max / (c->rope_beta_fast * 2.0f * 3.14159265358979323846f)))
               / (2.0f * logf(base));
    float hi_d = (dim * logf(c->rope_orig_max / (c->rope_beta_slow * 2.0f * 3.14159265358979323846f)))
               / (2.0f * logf(base));
    float low  = floorf(lo_d), high = ceilf(hi_d);
    if (low < 0) low = 0;
    if (high > dim - 1) high = dim - 1;
    float denom = (high == low) ? 0.001f : (high - low);

    float *inv = malloc(half * sizeof(float));
    for (int j = 0; j < half; j++) {
        float p        = (float)(2 * j) / (float)dim;
        float f_extra  = 1.0f / powf(base, p);            /* full-frequency */
        float f_inter  = f_extra / factor;                /* interpolated   */
        float ramp     = ((float)j - low) / denom;        /* 0..1 across range */
        if (ramp < 0) ramp = 0;
        if (ramp > 1) ramp = 1;
        float mask     = 1.0f - ramp;                     /* inv_freq_mask */
        inv[j] = f_inter * (1.0f - mask) + f_extra * mask;
    }
    return inv;
}

/* model_dir holds config.json, model.safetensors.index.json, and the shards. */
void build_transformer(Transformer *t, const char *model_dir) {
    char index_path[1024];
    snprintf(index_path, sizeof(index_path), "%s/model.safetensors.index.json", model_dir);
    /* 1. load config from config.json */
    t->config = config_from_json(model_dir);
    /* 2. mmap all shards */
    t->store = st_load_sharded(index_path, model_dir);
    /* 3. resolve tensor names → bf16_t* pointers */
    build_model_weights(t);
    /* 4. allocate run-state buffers */
    malloc_run_state(&t->state, &t->config);
    /* 5. precompute RoPE frequencies */
    t->rope_inv_freq = build_rope_inv_freq(&t->config);
}

void free_transformer(Transformer *t) {
    free(t->rope_inv_freq);
    free_run_state(&t->state);
    free_model_weights(&t->weights, t->config.n_layers);
    st_free(t->store);
}

/* ---------------------------------------------------------------------------
 * Unabsorbed (prefill) forward pass.
 * bf16 weights, fp32 compute; one stream, whole prompt in one call.
 * ------------------------------------------------------------------------- */

/* Validation dumps are compiled in only with -DMLA_ENABLE_DUMP (build with
 * `make DUMP=1`). By default DUMPING() is a compile-time 0, so every dump block
 * and its scratch is dead-code-eliminated — production builds carry no dump
 * code, branches, or I/O. Set the output dir at runtime via run_set_dump(). */
#ifdef MLA_ENABLE_DUMP
static const char *g_dump = NULL;
void run_set_dump(const char *dir) { g_dump = dir; }
#define DUMPING() (g_dump != NULL)

static void dump_bin(const char *name, const void *p, size_t n, size_t elem) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", g_dump, name);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "dump: cannot open %s\n", path); return; }
    fwrite(p, elem, n, f);
    fclose(f);
}
#define DUMP_F32(name, ptr, n) dump_bin(name, ptr, n, sizeof(float))
#define DUMP_I32(name, ptr, n) dump_bin(name, ptr, n, sizeof(int32_t))
#else
void run_set_dump(const char *dir) { (void)dir; }
#define DUMPING() 0
/* discard args (marks them used) so dead dump blocks don't warn */
#define DUMP_F32(name, ptr, n) ((void)(name), (void)(ptr), (void)(n))
#define DUMP_I32(name, ptr, n) ((void)(name), (void)(ptr), (void)(n))
#endif

/* y = (x / rms(x)) * w, over n elements (RMSNorm). */
static void rmsnorm(float *y, const float *x, const bf16_t *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * bf16_to_f32(w[i]);
}

/* y[d] = sum_i x[i] * W[d,i], W is bf16 row-major [d_out, n_in] (== x @ W.T). */
static void matmul(float *y, const float *x, const bf16_t *w, int d_out, int n_in) {
    for (int d = 0; d < d_out; d++) {
        const bf16_t *row = w + (size_t)d * n_in;
        float acc = 0.0f;
        for (int i = 0; i < n_in; i++) acc += x[i] * bf16_to_f32(row[i]);
        y[d] = acc;
    }
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

/* In-place RoPE on a rope_dim slice at `pos`. Two layouts, selected per model:
 *   interleaved==0 (dsv2): complex adjacent-pair — rotate (v[2j], v[2j+1]) in place.
 *   interleaved==1 (GLM):  even/odd pairs, output split-half
 *     [v0'..v31' | v32'..v63'] where v_j'=even_j*c-odd_j*s, v_{half+j}'=odd_j*c+even_j*s.
 * Both rotate by angle pos*inv_freq[j]; q and k share the layout, so it is
 * self-consistent in the score either way. */
static void rope_apply(float *v, int pos, const float *inv_freq, int rope_dim,
                       int interleaved) {
    int half = rope_dim / 2;
    if (!interleaved) {
        for (int j = 0; j < half; j++) {
            float ang = (float)pos * inv_freq[j];
            float c = cosf(ang), s = sinf(ang);
            float x0 = v[2 * j], x1 = v[2 * j + 1];
            v[2 * j]     = x0 * c - x1 * s;
            v[2 * j + 1] = x0 * s + x1 * c;
        }
        return;
    }
    float out[256];   /* qk_rope_head_dim <= 256 */
    for (int j = 0; j < half; j++) {
        float ang = (float)pos * inv_freq[j];
        float c = cosf(ang), s = sinf(ang);
        float even = v[2 * j], odd = v[2 * j + 1];
        out[j]        = even * c - odd * s;
        out[half + j] = odd  * c + even * s;
    }
    memcpy(v, out, rope_dim * sizeof(float));
}

/* Query projection for one token. dsv2: a single q_proj. GLM: a query LoRA
 * q = q_b_proj(rmsnorm(q_a_proj(x))). Writes q[n_heads*q_head_dim]. */
static void project_q(const Config *c, ModelWeights *w, RunState *s, int l,
                      const float *xn, float *q) {
    int q_dim = c->n_heads * (c->qk_nope_head_dim + c->qk_rope_head_dim);
    if (c->q_lora_rank > 0) {
        matmul(s->q_a, xn, w->q_a_proj[l], c->q_lora_rank, c->hidden_size);
        rmsnorm(s->q_a, s->q_a, w->q_a_layernorm[l], c->q_lora_rank, c->mla_norm_eps);
        matmul(q, s->q_a, w->q_b_proj[l], q_dim, c->q_lora_rank);
    } else {
        matmul(q, xn, w->q_proj[l], q_dim, c->hidden_size);
    }
}

/* SwiGLU MLP: dst = down( silu(gate·x) ⊙ (up·x) ). hb/hb2 are [inter] scratch. */
static void swiglu(float *dst, const float *x, const bf16_t *gate,
                   const bf16_t *up, const bf16_t *down,
                   int inter, int hidden, float *hb, float *hb2) {
    matmul(hb,  x, gate, inter, hidden);
    matmul(hb2, x, up,   inter, hidden);
    for (int i = 0; i < inter; i++) hb[i] = silu(hb[i]) * hb2[i];
    matmul(dst, hb, down, hidden, inter);
}

/* numerically-stable softmax over n elements (treats -inf as 0). */
static void softmax(float *x, int n) {
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = (x[i] == -INFINITY) ? 0.0f : expf(x[i] - mx);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* One token's FFN, shared by prefill and decode. xn is the post_attn_norm'd
 * hidden; writes out[hidden]. Dense SwiGLU for l < first_k_dense, else top-k
 * routed experts + shared expert. Router flavor is config-driven:
 *   dsv2: softmax scores, greedy top-k, weight = score, no norm, scale 1.
 *   GLM (noaux_tc): sigmoid scores, select by (score + e_score bias), weight =
 *     raw sigmoid score, normalize the top-k, then * routed_scaling.
 * (n_group/topk_group are 1 for GLM-4.7-Flash, so group-limited routing reduces
 * to a plain top-k — not implemented.) When topk_idx/topk_w are non-NULL the
 * chosen experts are returned; s->moe_logits is left holding the full scores. */
static void ffn_compute(const Config *c, ModelWeights *w, RunState *s, int l,
                        const float *xn, float *out, int *topk_idx, float *topk_w) {
    const int H = c->hidden_size;
    if (l < c->first_k_dense) {
        swiglu(out, xn, w->dense_gate[l], w->dense_up[l], w->dense_down[l],
               c->dense_inter_size, H, s->hb, s->hb2);
        return;
    }
    matmul(s->moe_logits, xn, w->moe_gate[l], c->n_routed_experts, H);
    if (c->router_sigmoid)
        for (int e = 0; e < c->n_routed_experts; e++)
            s->moe_logits[e] = 1.0f / (1.0f + expf(-s->moe_logits[e]));
    else
        softmax(s->moe_logits, c->n_routed_experts);

    const float *bias = w->moe_gate_bias[l];   /* GLM e_score_correction_bias (F32), else NULL */
    int  K = c->n_experts_per_tok;
    int  idx[16]; float wts[16];        /* K <= 16 */
    char used[1024] = {0};              /* n_routed <= 1024 */
    for (int r = 0; r < K; r++) {
        int best = -1; float bv = -INFINITY;
        for (int e = 0; e < c->n_routed_experts; e++) {
            if (used[e]) continue;
            float sel = s->moe_logits[e] + (bias ? bias[e] : 0.0f);
            if (sel > bv) { bv = sel; best = e; }
        }
        used[best] = 1; idx[r] = best; wts[r] = s->moe_logits[best];   /* raw score */
    }
    if (c->norm_topk) {
        float sum = 1e-20f;
        for (int r = 0; r < K; r++) sum += wts[r];
        for (int r = 0; r < K; r++) wts[r] /= sum;
    }
    for (int r = 0; r < K; r++) wts[r] *= c->routed_scaling;

    for (int i = 0; i < H; i++) out[i] = 0.0f;
    for (int r = 0; r < K; r++) {
        swiglu(s->expert_out, xn, w->expert_gate[l][idx[r]], w->expert_up[l][idx[r]],
               w->expert_down[l][idx[r]], c->moe_inter_size, H, s->hb, s->hb2);
        for (int i = 0; i < H; i++) out[i] += wts[r] * s->expert_out[i];
    }
    /* shared experts: one MLP of width n_shared * moe_inter */
    swiglu(s->expert_out, xn, w->shared_gate[l], w->shared_up[l], w->shared_down[l],
           c->n_shared_experts * c->moe_inter_size, H, s->hb, s->hb2);
    for (int i = 0; i < H; i++) out[i] += s->expert_out[i];
    if (topk_idx) for (int r = 0; r < K; r++) { topk_idx[r] = idx[r]; topk_w[r] = wts[r]; }
}

/* Whole-prompt prefill. Returns logits at the LAST position (RunState.logits).
 * When built with -DMLA_ENABLE_DUMP and a dump dir is set, also writes the
 * oracle-named intermediates for validation. */
float *forward_unabsorbed(Transformer *t, const int *tokens, int n_prompt) {
    const Config *c = &t->config;
    ModelWeights *w = &t->weights;
    RunState     *s = &t->state;
    const float  *inv_freq = t->rope_inv_freq;

    const int H   = c->hidden_size;
    const int NH  = c->n_heads;
    const int QKN = c->qk_nope_head_dim, QKR = c->qk_rope_head_dim;
    const int QHD = QKN + QKR;
    const int VHD = c->v_head_dim;
    const int KVL = c->kv_lora_rank;
    const int KVD = KVL + QKR;
    const int VOC = c->vocab_size;
    const float scale = c->softmax_scale, eps = c->rms_eps;
    const size_t kv_stride = (size_t)(QKN + VHD) * KVL;   /* per-head kv_b stride */
    const size_t cache_row = (size_t)c->max_seq_len * KVD;

    /* residual stream for all prompt positions */
    float *xs = malloc((size_t)n_prompt * H * sizeof(float));
    /* per-layer scratch (reused across layers) */
    float *xb    = malloc(H * sizeof(float));
    float *comp  = malloc(KVD * sizeof(float));
    float *qall  = malloc((size_t)n_prompt * NH * QHD * sizeof(float));
    float *knope = malloc((size_t)NH * n_prompt * QKN * sizeof(float));   /* [h][k][d] */
    float *value = malloc((size_t)NH * n_prompt * VHD * sizeof(float));   /* [h][k][d] */
    float *scr   = malloc((size_t)NH * n_prompt * n_prompt * sizeof(float)); /* [h][q][k] scores */
    float *wgt   = malloc((size_t)NH * n_prompt * n_prompt * sizeof(float)); /* [h][q][k] weights */
    float *ctx   = malloc((size_t)NH * VHD * sizeof(float));
    float *ao    = malloc((size_t)n_prompt * H * sizeof(float));   /* attn module out */
    float *mo    = malloc((size_t)n_prompt * H * sizeof(float));   /* mlp module out */
    /* hidden_states snapshot [n_layers+1][n_prompt][H] (dump only) */
    float *hs = DUMPING() ? malloc((size_t)(c->n_layers + 1) * n_prompt * H * sizeof(float)) : NULL;

    /* embed */
    for (int p = 0; p < n_prompt; p++)
        for (int i = 0; i < H; i++)
            xs[p * H + i] = bf16_to_f32(w->embed_tokens[(size_t)tokens[p] * H + i]);
    if (hs) memcpy(hs, xs, (size_t)n_prompt * H * sizeof(float));

    for (int l = 0; l < c->n_layers; l++) {
        float *kv_l = &s->kv_cache[(size_t)l * cache_row];

        /* ---- per-position projections: q (roped), c_kv, k_pe (roped, cached) ---- */
        for (int p = 0; p < n_prompt; p++) {
            rmsnorm(xb, &xs[p * H], w->input_layernorm[l], H, eps);
            project_q(c, w, s, l, xb, &qall[(size_t)p * NH * QHD]);
            for (int h = 0; h < NH; h++)
                rope_apply(&qall[((size_t)p * NH + h) * QHD + QKN], p, inv_freq, QKR,
                           c->rope_interleaved);

            matmul(comp, xb, w->kv_a_proj[l], KVD, H);
            float *cache_p = &kv_l[(size_t)p * KVD];
            rmsnorm(cache_p, comp, w->kv_a_layernorm[l], KVL, c->mla_norm_eps);  /* c_kv */
            memcpy(cache_p + KVL, comp + KVL, QKR * sizeof(float));  /* k_pe (raw) */
            rope_apply(cache_p + KVL, p, inv_freq, QKR, c->rope_interleaved); /* roped */
        }

        /* ---- decompress per-head K_nope and V for every key ---- */
        for (int h = 0; h < NH; h++) {
            const bf16_t *WUK = w->W_UK[l] + (size_t)h * kv_stride;
            const bf16_t *WUV = w->W_UV[l] + (size_t)h * kv_stride;
            for (int k = 0; k < n_prompt; k++) {
                const float *ckv = &kv_l[(size_t)k * KVD];
                matmul(&knope[((size_t)h * n_prompt + k) * QKN], ckv, WUK, QKN, KVL);
                matmul(&value[((size_t)h * n_prompt + k) * VHD], ckv, WUV, VHD, KVL);
            }
        }

        /* ---- attention: per query, per head ---- */
        for (int q = 0; q < n_prompt; q++) {
            for (int h = 0; h < NH; h++) {
                const float *qnope = &qall[((size_t)q * NH + h) * QHD];
                const float *qpe   = qnope + QKN;
                float *row = &scr[((size_t)h * n_prompt + q) * n_prompt];
                for (int k = 0; k < n_prompt; k++) {
                    if (k > q) { row[k] = -INFINITY; continue; }   /* causal */
                    const float *knope_hk = &knope[((size_t)h * n_prompt + k) * QKN];
                    const float *kpe_k    = &kv_l[(size_t)k * KVD + KVL];
                    float dot = 0.0f;
                    for (int d = 0; d < QKN; d++) dot += qnope[d] * knope_hk[d];
                    for (int d = 0; d < QKR; d++) dot += qpe[d]   * kpe_k[d];
                    row[k] = dot * scale;
                }
                float *wrow = &wgt[((size_t)h * n_prompt + q) * n_prompt];
                memcpy(wrow, row, n_prompt * sizeof(float));
                softmax(wrow, n_prompt);

                float *ctx_h = &ctx[(size_t)h * VHD];
                for (int d = 0; d < VHD; d++) ctx_h[d] = 0.0f;
                for (int k = 0; k <= q; k++) {
                    float a = wrow[k];
                    const float *val_hk = &value[((size_t)h * n_prompt + k) * VHD];
                    for (int d = 0; d < VHD; d++) ctx_h[d] += a * val_hk[d];
                }
            }
            matmul(&ao[(size_t)q * H], ctx, w->o_proj[l], H, NH * VHD);
        }
        for (int p = 0; p < n_prompt; p++)
            for (int i = 0; i < H; i++) xs[p * H + i] += ao[p * H + i];

        /* ---- FFN: dense for l < first_k_dense, else MoE (shared helper) ---- */
        int dump_router = DUMPING() && (l == c->first_k_dense);
        for (int p = 0; p < n_prompt; p++) {
            rmsnorm(xb, &xs[p * H], w->post_attn_norm[l], H, eps);
            int ti_p[16]; float tw_p[16];
            ffn_compute(c, w, s, l, xb, &mo[(size_t)p * H],
                        dump_router ? ti_p : NULL, dump_router ? tw_p : NULL);
            /* dump first MoE layer's router (oracle: prefill_moe1_*) */
            if (dump_router) {
                static float rs[8 * 1024]; static int ti[8 * 16]; static float tw[8 * 16];
                int K = c->n_experts_per_tok;
                memcpy(&rs[(size_t)p * c->n_routed_experts], s->moe_logits,
                       c->n_routed_experts * sizeof(float));   /* router softmax */
                for (int r = 0; r < K; r++) { ti[p * K + r] = ti_p[r]; tw[p * K + r] = tw_p[r]; }
                if (p == n_prompt - 1) {
                    DUMP_F32("prefill_moe1_router_scores", rs, (size_t)n_prompt * c->n_routed_experts);
                    DUMP_I32("prefill_moe1_topk_idx", ti, (size_t)n_prompt * K);
                    DUMP_F32("prefill_moe1_topk_w",   tw, (size_t)n_prompt * K);
                }
            }
        }
        for (int p = 0; p < n_prompt; p++)
            for (int i = 0; i < H; i++) xs[p * H + i] += mo[p * H + i];

        /* ---- per-layer dumps ---- */
        if (DUMPING()) {
            char nm[64];
            snprintf(nm, sizeof(nm), "prefill_layer%02d_attn_out", l);
            DUMP_F32(nm, ao, (size_t)n_prompt * H);
            snprintf(nm, sizeof(nm), "prefill_layer%02d_mlp_out", l);
            DUMP_F32(nm, mo, (size_t)n_prompt * H);
            memcpy(&hs[(size_t)(l + 1) * n_prompt * H], xs, (size_t)n_prompt * H * sizeof(float));
            if (l == 0) {
                /* layer-0 MLA internals, in the oracle's dump orders.
                 * q_nope/q_pe are [n_heads, seq, dim] (heads-first, like k_nope). */
                float *t1 = malloc((size_t)NH * n_prompt * QKN * sizeof(float));
                for (int h = 0; h < NH; h++)
                    for (int p = 0; p < n_prompt; p++)
                        memcpy(&t1[((size_t)h * n_prompt + p) * QKN],
                               &qall[((size_t)p * NH + h) * QHD], QKN * sizeof(float));
                DUMP_F32("prefill_layer0_q_nope", t1, (size_t)NH * n_prompt * QKN);
                float *t2 = malloc((size_t)NH * n_prompt * QKR * sizeof(float));
                for (int h = 0; h < NH; h++)
                    for (int p = 0; p < n_prompt; p++)
                        memcpy(&t2[((size_t)h * n_prompt + p) * QKR],
                               &qall[((size_t)p * NH + h) * QHD + QKN], QKR * sizeof(float));
                DUMP_F32("prefill_layer0_q_pe", t2, (size_t)NH * n_prompt * QKR);
                float *t3 = malloc((size_t)n_prompt * KVL * sizeof(float));
                float *t4 = malloc((size_t)n_prompt * QKR * sizeof(float));
                for (int p = 0; p < n_prompt; p++) {    /* c_kv [seq,kv_lora], k_pe [seq,qk_rope] */
                    memcpy(&t3[(size_t)p * KVL], &kv_l[(size_t)p * KVD], KVL * sizeof(float));
                    memcpy(&t4[(size_t)p * QKR], &kv_l[(size_t)p * KVD + KVL], QKR * sizeof(float));
                }
                DUMP_F32("prefill_layer0_c_kv", t3, (size_t)n_prompt * KVL);
                DUMP_F32("prefill_layer0_k_pe", t4, (size_t)n_prompt * QKR);
                DUMP_F32("prefill_layer0_k_nope", knope, (size_t)NH * n_prompt * QKN);
                DUMP_F32("prefill_layer0_value", value, (size_t)NH * n_prompt * VHD);
                DUMP_F32("prefill_layer0_attn_scores", scr, (size_t)NH * n_prompt * n_prompt);
                DUMP_F32("prefill_layer0_attn_weights", wgt, (size_t)NH * n_prompt * n_prompt);
                free(t1); free(t2); free(t3); free(t4);
            }
        }
    }

    /* final norm + lm_head */
    if (DUMPING()) {
        float *logits_all = malloc((size_t)n_prompt * VOC * sizeof(float));
        for (int p = 0; p < n_prompt; p++) {
            rmsnorm(xb, &xs[p * H], w->norm, H, eps);
            /* HF's final hidden_states entry is post-norm: overwrite hs[n_layers] */
            memcpy(&hs[((size_t)c->n_layers * n_prompt + p) * H], xb, H * sizeof(float));
            matmul(&logits_all[(size_t)p * VOC], xb, w->lm_head, VOC, H);
        }
        DUMP_F32("prefill_logits", logits_all, (size_t)n_prompt * VOC);
        DUMP_F32("prefill_hidden_states", hs, (size_t)(c->n_layers + 1) * n_prompt * H);
        memcpy(s->logits, &logits_all[(size_t)(n_prompt - 1) * VOC], VOC * sizeof(float));
        free(logits_all);
    } else {
        rmsnorm(xb, &xs[(size_t)(n_prompt - 1) * H], w->norm, H, eps);
        matmul(s->logits, xb, w->lm_head, VOC, H);
    }

    free(xs); free(xb); free(comp); free(qall); free(knope); free(value);
    free(scr); free(wgt); free(ctx); free(ao); free(mo); free(hs);
    return s->logits;
}

/* Single-token decode via the ABSORBED MLA path. Reads the latent KV cache
 * filled by prefill (and prior decode steps), appends this position's c_kv/k_pe,
 * and attends in latent space: W_UK is folded into the query and W_UV into the
 * output, so no per-head K/V is ever materialized. Mathematically identical to
 * forward_unabsorbed for the same inputs; cheaper at q_len=1. Returns logits.
 * When built with -DMLA_ENABLE_DUMP and a dump dir is set, writes the layer-0
 * decode_* internals (oracle-named). */
float *forward_absorbed(Transformer *t, int token, int pos) {
    const Config *c = &t->config;
    ModelWeights *w = &t->weights;
    RunState     *s = &t->state;
    const float  *inv_freq = t->rope_inv_freq;

    const int H   = c->hidden_size;
    const int NH  = c->n_heads;
    const int QKN = c->qk_nope_head_dim, QKR = c->qk_rope_head_dim;
    const int QHD = QKN + QKR;
    const int VHD = c->v_head_dim;
    const int KVL = c->kv_lora_rank;
    const int KVD = KVL + QKR;
    const int VOC = c->vocab_size;
    const float scale = c->softmax_scale, eps = c->rms_eps;
    const size_t kv_stride = (size_t)(QKN + VHD) * KVL;
    const size_t cache_row = (size_t)c->max_seq_len * KVD;
    const int kv_len = pos + 1;                /* keys: [0..pos] inclusive */

    float *x    = malloc(H * sizeof(float));
    float *xb   = malloc(H * sizeof(float));
    float *comp = malloc(KVD * sizeof(float));
    float *q    = malloc((size_t)NH * QHD * sizeof(float));
    float *qabs = malloc(KVL * sizeof(float));            /* q_absorbed (per head) */
    float *score= malloc(kv_len * sizeof(float));
    float *clat = malloc(KVL * sizeof(float));            /* latent context (per head) */
    float *ctx  = malloc((size_t)NH * VHD * sizeof(float));
    float *ao   = malloc(H * sizeof(float));
    float *mo   = malloc(H * sizeof(float));
    /* layer-0 decode dump scratch (heads-major, like the oracle) */
    float *d_qabs = DUMPING() ? malloc((size_t)NH * KVL * sizeof(float)) : NULL;
    float *d_clat = DUMPING() ? malloc((size_t)NH * KVL * sizeof(float)) : NULL;
    float *d_attn = DUMPING() ? malloc((size_t)NH * kv_len * sizeof(float)) : NULL;

    for (int i = 0; i < H; i++) x[i] = bf16_to_f32(w->embed_tokens[(size_t)token * H + i]);

    for (int l = 0; l < c->n_layers; l++) {
        float *kv_l = &s->kv_cache[(size_t)l * cache_row];

        rmsnorm(xb, x, w->input_layernorm[l], H, eps);
        project_q(c, w, s, l, xb, q);
        for (int h = 0; h < NH; h++)
            rope_apply(&q[(size_t)h * QHD + QKN], pos, inv_freq, QKR, c->rope_interleaved);

        /* append this position's latent KV */
        matmul(comp, xb, w->kv_a_proj[l], KVD, H);
        float *cache_p = &kv_l[(size_t)pos * KVD];
        rmsnorm(cache_p, comp, w->kv_a_layernorm[l], KVL, c->mla_norm_eps);
        memcpy(cache_p + KVL, comp + KVL, QKR * sizeof(float));
        rope_apply(cache_p + KVL, pos, inv_freq, QKR, c->rope_interleaved);

        for (int h = 0; h < NH; h++) {
            const bf16_t *WUK = w->W_UK[l] + (size_t)h * kv_stride;
            const bf16_t *WUV = w->W_UV[l] + (size_t)h * kv_stride;
            const float  *qnope = &q[(size_t)h * QHD];
            const float  *qpe   = qnope + QKN;

            /* q_absorbed[r] = sum_d q_nope[d] * W_UK[d,r]  (fold W_UK into query) */
            for (int r = 0; r < KVL; r++) qabs[r] = 0.0f;
            for (int d = 0; d < QKN; d++) {
                float qd = qnope[d];
                const bf16_t *row = WUK + (size_t)d * KVL;
                for (int r = 0; r < KVL; r++) qabs[r] += qd * bf16_to_f32(row[r]);
            }
            /* scores in latent space: q_absorbed·c_kv + q_pe·k_pe */
            for (int k = 0; k < kv_len; k++) {
                const float *ckv  = &kv_l[(size_t)k * KVD];
                const float *kpe  = ckv + KVL;
                float sn = 0.0f, sp = 0.0f;
                for (int r = 0; r < KVL; r++) sn += qabs[r] * ckv[r];
                for (int d = 0; d < QKR; d++) sp += qpe[d]  * kpe[d];
                score[k] = (sn + sp) * scale;
            }
            softmax(score, kv_len);
            /* latent context, then up-project with W_UV */
            for (int r = 0; r < KVL; r++) clat[r] = 0.0f;
            for (int k = 0; k < kv_len; k++) {
                float a = score[k];
                const float *ckv = &kv_l[(size_t)k * KVD];
                for (int r = 0; r < KVL; r++) clat[r] += a * ckv[r];
            }
            matmul(&ctx[(size_t)h * VHD], clat, WUV, VHD, KVL);

            if (DUMPING() && l == 0) {
                memcpy(&d_qabs[(size_t)h * KVL], qabs, KVL * sizeof(float));
                memcpy(&d_clat[(size_t)h * KVL], clat, KVL * sizeof(float));
                memcpy(&d_attn[(size_t)h * kv_len], score, kv_len * sizeof(float));
            }
        }
        matmul(ao, ctx, w->o_proj[l], H, NH * VHD);
        for (int i = 0; i < H; i++) x[i] += ao[i];

        rmsnorm(xb, x, w->post_attn_norm[l], H, eps);
        ffn_compute(c, w, s, l, xb, mo, NULL, NULL);
        for (int i = 0; i < H; i++) x[i] += mo[i];

        if (DUMPING() && l == 0) {
            DUMP_F32("decode_layer0_q_absorbed", d_qabs, (size_t)NH * KVL);
            DUMP_F32("decode_layer0_ctx_latent", d_clat, (size_t)NH * KVL);
            DUMP_F32("decode_layer0_attn_weights", d_attn, (size_t)NH * kv_len);
            DUMP_F32("decode_layer0_attn_out", ao, H);
            /* the full latent cache (prefill + this step), heads-shared */
            float *cc = malloc((size_t)kv_len * KVL * sizeof(float));
            float *kk = malloc((size_t)kv_len * QKR * sizeof(float));
            for (int k = 0; k < kv_len; k++) {
                memcpy(&cc[(size_t)k * KVL], &kv_l[(size_t)k * KVD], KVL * sizeof(float));
                memcpy(&kk[(size_t)k * QKR], &kv_l[(size_t)k * KVD + KVL], QKR * sizeof(float));
            }
            DUMP_F32("decode_layer0_c_kv_cache", cc, (size_t)kv_len * KVL);
            DUMP_F32("decode_layer0_k_pe_cache", kk, (size_t)kv_len * QKR);
            free(cc); free(kk);
        }
    }

    rmsnorm(xb, x, w->norm, H, eps);
    matmul(s->logits, xb, w->lm_head, VOC, H);
    if (DUMPING()) DUMP_F32("decode_logits", s->logits, VOC);

    free(x); free(xb); free(comp); free(q); free(qabs); free(score);
    free(clat); free(ctx); free(ao); free(mo);
    free(d_qabs); free(d_clat); free(d_attn);
    return s->logits;
}

/* greedy argmax over logits */
int sample(float *logits, int vocab_size) {
    int best = 0; float bv = logits[0];
    for (int i = 1; i < vocab_size; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

/* Read up to max int32 tokens from a raw little-endian .i32.bin file. */
static int read_tokens_i32(const char *path, int *out, int max) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open tokens file: %s\n", path); exit(1); }
    int32_t buf[4096];
    size_t n = fread(buf, sizeof(int32_t), max < 4096 ? max : 4096, f);
    fclose(f);
    for (size_t i = 0; i < n; i++) out[i] = buf[i];
    return (int)n;
}

/* Two-path generation: one unabsorbed prefill over the prompt, then a greedy
 * absorbed-decode loop. Prints generated token ids (no detokenizer in C yet). */
static void generate(Transformer *t, const int *prompt, int n_prompt, int max_new) {
    float *logits = forward_unabsorbed(t, prompt, n_prompt);
    int tok = sample(logits, t->config.vocab_size);
    int pos = n_prompt;
    printf("generated:");
    for (int i = 0; i < max_new; i++) {
        printf(" %d", tok);
        logits = forward_absorbed(t, tok, pos);
        tok = sample(logits, t->config.vocab_size);
        pos++;
    }
    printf(" %d\n", tok);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <model_dir> [tokens.i32.bin] [dump_dir]\n"
            "  model_dir: holds config.json, model.safetensors.index.json, shards\n"
            "             (model family auto-detected from config.json)\n"
            "  no tokens file  -> weight-load smoke test\n"
            "  tokens file     -> prefill + greedy absorbed decode (prints token ids)\n"
            "  + dump_dir      -> prefill + ONE decode step, writing oracle-named\n"
            "                     prefill_*/decode_* intermediates for validation\n", argv[0]);
        return 1;
    }
    const char *model_dir   = argv[1];
    const char *tokens_path = (argc > 2) ? argv[2] : NULL;
    const char *dump_dir    = (argc > 3) ? argv[3] : NULL;
#ifndef MLA_ENABLE_DUMP
    if (dump_dir)
        fprintf(stderr, "note: dump_dir given but dumps not compiled in "
                        "(rebuild with `make DUMP=1`); running without dumps.\n");
#endif

    Transformer t;
    build_transformer(&t, model_dir);
    printf("Loaded %zu tensors\n", st_count(t.store));
    printf("Config: n_layers=%d hidden=%d vocab=%d\n",
           t.config.n_layers, t.config.hidden_size, t.config.vocab_size);

    if (!tokens_path) {
        printf("weights loaded; pass a tokens file to run prefill.\n");
        free_transformer(&t);
        return 0;
    }

    int tokens[4096];
    int n_prompt = read_tokens_i32(tokens_path, tokens, 4096);
    printf("Prompt: %d tokens [", n_prompt);
    for (int i = 0; i < n_prompt; i++) printf("%s%d", i ? ", " : "", tokens[i]);
    printf("]\n");

    if (dump_dir) {
        /* validation scenario: prefill, then exactly one absorbed-decode step
         * (mirrors gen_oracle: feed the prefill argmax at position n_prompt). */
        run_set_dump(dump_dir);
        float *logits = forward_unabsorbed(&t, tokens, n_prompt);
        int next = sample(logits, t.config.vocab_size);
        printf("prefill argmax = %d  logit=%.6f\n", next, logits[next]);
        float *dlog = forward_absorbed(&t, next, n_prompt);
        int next2 = sample(dlog, t.config.vocab_size);
        printf("decode  argmax = %d  logit=%.6f\n", next2, dlog[next2]);
    } else {
        generate(&t, tokens, n_prompt, 16);
    }

    free_transformer(&t);
    return 0;
}
