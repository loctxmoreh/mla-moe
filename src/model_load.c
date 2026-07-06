/* Model loading: config.json -> Config, safetensors -> ModelWeights (zero-copy
 * pointers into the mmaps), run-state allocation, and YaRN inv_freq precompute.
 * Split out of run.c so that file holds only the forward-pass math. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "model.h"
#include "cJSON.h"

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

/* ---- config.json reader (vendored cJSON) --------------------------------- */

static char *read_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "read %s\n", path); exit(1); }
    buf[n] = '\0'; fclose(f);
    return buf;
}

/* typed getters over a cJSON object; all tolerate missing keys / null / NULL obj */
static int cfg_int(const cJSON *o, const char *k, int dflt) {
    const cJSON *v = cJSON_GetObjectItemCaseSensitive(o, k);
    return cJSON_IsNumber(v) ? (int)v->valuedouble : dflt;
}
static double cfg_double(const cJSON *o, const char *k, double dflt) {
    const cJSON *v = cJSON_GetObjectItemCaseSensitive(o, k);
    return cJSON_IsNumber(v) ? v->valuedouble : dflt;
}
static int cfg_bool(const cJSON *o, const char *k, int dflt) {
    const cJSON *v = cJSON_GetObjectItemCaseSensitive(o, k);
    return cJSON_IsBool(v) ? cJSON_IsTrue(v) : dflt;
}
static int cfg_str_is(const cJSON *o, const char *k, const char *val) {
    const cJSON *v = cJSON_GetObjectItemCaseSensitive(o, k);
    return cJSON_IsString(v) && v->valuestring && strcmp(v->valuestring, val) == 0;
}

/* Cap on the latent KV cache length: the lower of the two target models'
 * max_position_embeddings (dsv2lite 163840, glm47 202752). Prompt + generated
 * tokens must fit under this. The cache is calloc'd at this length but pages are
 * lazily backed, so unused positions cost no resident memory. */
#define KV_CACHE_CAP 163840

/* Build Config from <model_dir>/config.json. Model-family behavior is derived,
 * not hardcoded: rope layout from model_type, router flavor from topk_method,
 * YaRN params from rope_scaling (absent/null => plain rope). */
static Config config_from_json(const char *model_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    char *text = read_text_file(path);
    cJSON *r = cJSON_Parse(text);
    if (!r) { fprintf(stderr, "config_from_json: parse error in %s\n", path); exit(1); }

    Config c = {0};
    c.n_layers          = cfg_int(r, "num_hidden_layers", 0);
    c.hidden_size       = cfg_int(r, "hidden_size", 0);
    c.n_heads           = cfg_int(r, "num_attention_heads", 0);
    c.n_kv_heads        = c.n_heads;                       /* MLA: no GQA */
    c.q_lora_rank       = cfg_int(r, "q_lora_rank", 0);    /* null => 0 (no q-LoRA) */
    c.first_k_dense     = cfg_int(r, "first_k_dense_replace", 0);
    c.kv_lora_rank      = cfg_int(r, "kv_lora_rank", 0);
    c.qk_rope_head_dim  = cfg_int(r, "qk_rope_head_dim", 0);
    c.qk_nope_head_dim  = cfg_int(r, "qk_nope_head_dim", 0);
    c.v_head_dim        = cfg_int(r, "v_head_dim", 0);
    c.n_routed_experts  = cfg_int(r, "n_routed_experts", 0);
    c.n_shared_experts  = cfg_int(r, "n_shared_experts", 0);
    c.n_experts_per_tok = cfg_int(r, "num_experts_per_tok", 0);
    c.dense_inter_size  = cfg_int(r, "intermediate_size", 0);
    c.moe_inter_size    = cfg_int(r, "moe_intermediate_size", 0);
    c.vocab_size        = cfg_int(r, "vocab_size", 0);
    int maxpos          = cfg_int(r, "max_position_embeddings", KV_CACHE_CAP);
    c.max_seq_len       = maxpos < KV_CACHE_CAP ? maxpos : KV_CACHE_CAP;
    c.rms_eps           = (float)cfg_double(r, "rms_norm_eps", 1e-5);
    c.mla_norm_eps      = 1e-6f;   /* q_a/kv_a RMSNorm class default — not in config.json */
    c.softmax_scale     = 1.0f / sqrtf((float)(c.qk_nope_head_dim + c.qk_rope_head_dim));
    c.rope_theta        = (float)cfg_double(r, "rope_theta", 10000.0);
    /* rope_scaling (YaRN) is an object when present, JSON null otherwise =>
     * plain rope (factor 1). */
    const cJSON *rs = cJSON_GetObjectItemCaseSensitive(r, "rope_scaling");
    if (!cJSON_IsObject(rs)) rs = NULL;
    c.rope_factor    = rs ? (float)cfg_double(rs, "factor", 1.0) : 1.0f;
    c.rope_beta_fast = rs ? (float)cfg_double(rs, "beta_fast", 32.0) : 32.0f;
    c.rope_beta_slow = rs ? (float)cfg_double(rs, "beta_slow", 1.0) : 1.0f;
    c.rope_orig_max  = rs ? (float)cfg_double(rs, "original_max_position_embeddings", 4096.0) : 4096.0f;
    c.rope_interleaved  = cfg_str_is(r, "model_type", "glm4_moe_lite");  /* else complex */
    c.router_sigmoid    = cfg_str_is(r, "topk_method", "noaux_tc");      /* else softmax-greedy */
    c.norm_topk         = cfg_bool(r, "norm_topk_prob", 0);
    c.routed_scaling    = (float)cfg_double(r, "routed_scaling_factor", 1.0);

    cJSON_Delete(r);
    free(text);

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
    /* 6. tokenizer (optional — absent for raw-token-id / oracle workflows) */
    t->tokenizer = tokenizer_load(model_dir);
}

void free_transformer(Transformer *t) {
    tokenizer_free(t->tokenizer);
    free(t->rope_inv_freq);
    free_run_state(&t->state);
    free_model_weights(&t->weights, t->config.n_layers);
    st_free(t->store);
}
