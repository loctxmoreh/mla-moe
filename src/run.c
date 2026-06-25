/* MLA-MOE inference entry point; structure follows llama2.c.
 * Phase 0: forward()/sample() are stubs; the loader + teardown are real. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tensor.h"
#include "safetensors_loader.h"
#include "model_config.h"

/* Mirrors llama2.c's Config; hardcoded from model_config.h in Phase 0. */
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
    int   vocab_size;
    int   max_seq_len;
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
    bf16_t **moe_gate_bias;  /* [n_layers]: mlp.gate.e_score_correction_bias — GLM only, else NULL */

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
 * never copies tensor data. The real work of Phase 0. */
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
    w->moe_gate_bias   = malloc(c->n_layers * sizeof(bf16_t*));
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

            w->moe_gate[l] = w->moe_gate_bias[l] = NULL;
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
        const Tensor *bias = st_get(s, name);
        w->moe_gate_bias[l] = bias ? bias->data : NULL;

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
    int inter    = 1408;    /* hardcoded for Phase 0; read from config later */

    s->x          = calloc(h, sizeof(float));
    s->xb         = calloc(h, sizeof(float));
    s->xb2        = calloc(h, sizeof(float));
    s->q          = calloc(q_dim, sizeof(float));
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
    free(s->q);   free(s->c_kv); free(s->att);
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

/* Hardcoded per-model config for Phase 0 (config.json-driven later). */
static Config config_for(const char *model) {
    if (strcmp(model, "dsv2lite") == 0) {
        return (Config){
            .n_layers           = DSV2LITE_N_LAYERS,
            .hidden_size        = DSV2LITE_HIDDEN,
            .n_heads            = DSV2LITE_N_HEADS,
            .n_kv_heads         = DSV2LITE_N_KV_HEADS,
            .q_lora_rank        = DSV2LITE_Q_LORA_RANK,
            .first_k_dense      = DSV2LITE_FIRST_K_DENSE,
            .kv_lora_rank       = DSV2LITE_KV_LORA_RANK,
            .qk_rope_head_dim   = DSV2LITE_QK_ROPE_HEAD_DIM,
            .qk_nope_head_dim   = DSV2LITE_QK_NOPE_HEAD_DIM,
            .v_head_dim         = DSV2LITE_V_HEAD_DIM,
            .n_routed_experts   = DSV2LITE_N_ROUTED_EXPERTS,
            .n_shared_experts   = DSV2LITE_N_SHARED_EXPERTS,
            .n_experts_per_tok  = DSV2LITE_N_EXPERTS_PER_TOK,
            .vocab_size         = DSV2LITE_VOCAB_SIZE,
            .max_seq_len        = DSV2LITE_MAX_SEQ_LEN,
        };
    }
    if (strcmp(model, "glm47") == 0) {
        return (Config){
            .n_layers           = GLM47_N_LAYERS,
            .hidden_size        = GLM47_HIDDEN,
            .n_heads            = GLM47_N_HEADS,
            .n_kv_heads         = GLM47_N_KV_HEADS,
            .q_lora_rank        = GLM47_Q_LORA_RANK,
            .first_k_dense      = GLM47_FIRST_K_DENSE,
            .kv_lora_rank       = GLM47_KV_LORA_RANK,
            .qk_rope_head_dim   = GLM47_QK_ROPE_HEAD_DIM,
            .qk_nope_head_dim   = GLM47_QK_NOPE_HEAD_DIM,
            .v_head_dim         = GLM47_V_HEAD_DIM,
            .n_routed_experts   = GLM47_N_ROUTED_EXPERTS,
            .n_shared_experts   = GLM47_N_SHARED_EXPERTS,
            .n_experts_per_tok  = GLM47_N_EXPERTS_PER_TOK,
            .vocab_size         = GLM47_VOCAB_SIZE,
            .max_seq_len        = GLM47_MAX_SEQ_LEN,
        };
    }
    fprintf(stderr, "unknown model '%s' (expected: dsv2lite | glm47)\n", model);
    exit(1);
}

void build_transformer(Transformer *t, const char *model,
                       const char *index_path, const char *shard_dir) {
    /* 1. load config (hardcoded per model for Phase 0) */
    t->config = config_for(model);
    /* 2. mmap all shards */
    t->store = st_load_sharded(index_path, shard_dir);
    /* 3. resolve tensor names → bf16_t* pointers */
    build_model_weights(t);
    /* 4. allocate run-state buffers */
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
    free_run_state(&t->state);
    free_model_weights(&t->weights, t->config.n_layers);
    st_free(t->store);
}

/* Inference stubs — filled in Phase 1+. */

/* One transformer step; returns logits (owned by RunState). */
float *forward(Transformer *t, int token, int pos) {
    (void)t; (void)token; (void)pos;
    fprintf(stderr, "forward(): not yet implemented\n");
    exit(1);
}

int sample(float *logits, int vocab_size) {
    (void)logits; (void)vocab_size;
    fprintf(stderr, "sample(): not yet implemented\n");
    exit(1);
}

/* Mirrors llama2.c's generate() loop; locks in the structure for Phase 1. */
static void generate(Transformer *t, int *prompt_tokens, int n_prompt,
                     int max_new_tokens) {
    int token = prompt_tokens[0];
    int pos   = 0;

    while (pos < n_prompt + max_new_tokens - 1) {
        float *logits = forward(t, token, pos);  /* stub exits here in Phase 0 */
        int next;
        if (pos < n_prompt - 1) {
            next = prompt_tokens[pos + 1];       /* teacher-force prompt tokens */
        } else {
            next = sample(logits, t->config.vocab_size);
        }
        token = next;
        pos++;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <model> <index_json> <shard_dir>\n"
            "  model: dsv2lite | glm47\n"
            "Phase 0: loads weights and exits cleanly.\n", argv[0]);
        return 1;
    }
    const char *model      = argv[1];
    const char *index_path = argv[2];
    const char *shard_dir  = argv[3];

    Transformer t;
    build_transformer(&t, model, index_path, shard_dir);

    printf("Loaded %zu tensors\n", st_count(t.store));
    printf("Config: n_layers=%d hidden=%d vocab=%d\n",
           t.config.n_layers, t.config.hidden_size, t.config.vocab_size);

    /* Smoke test: spot-check a few key tensors (query path depends on the model). */
    printf("embed_tokens[0] = %f\n", bf16_to_f32(t.weights.embed_tokens[0]));
    if (t.config.q_lora_rank > 0)
        printf("q_a_proj[0][0]  = %f\n", bf16_to_f32(t.weights.q_a_proj[0][0]));
    else
        printf("q_proj[0][0]    = %f\n", bf16_to_f32(t.weights.q_proj[0][0]));
    printf("W_UK[0][0]      = %f\n", bf16_to_f32(t.weights.W_UK[0][0]));
    printf("W_UV[0][0]      = %f\n", bf16_to_f32(t.weights.W_UV[0][0]));
    printf("lm_head[0]      = %f\n", bf16_to_f32(t.weights.lm_head[0]));

    (void)generate;  /* unused in Phase 0 */
    free_transformer(&t);
    printf("Phase 0 complete — weights loaded and freed cleanly.\n");
    return 0;
}
