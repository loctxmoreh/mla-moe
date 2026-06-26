/* Per-model dims, hardcoded (config.json-driven later). Values are
 * from each upstream config.json, verified against the on-disk weights.
 *
 * Layout caveats vs the idealized MLA+MoE:
 *   - DeepSeek-V2-Lite: q_lora_rank=null -> single q_proj, no q_a/q_b/q_a_layernorm.
 *     GLM-4.7-Flash uses a query LoRA (768) so it does have them.
 *   - first_k_dense_replace=1 (both): layer 0 is a dense FFN, not MoE.
 */
#pragma once

/* DeepSeek-V2-Lite (15.7B / 2.4B active) — primary target */
#define DSV2LITE_N_LAYERS            27
#define DSV2LITE_HIDDEN              2048
#define DSV2LITE_N_HEADS             16
#define DSV2LITE_N_KV_HEADS          16
#define DSV2LITE_KV_LORA_RANK        512
#define DSV2LITE_Q_LORA_RANK         0      /* null upstream: q_proj is not LoRA'd */
#define DSV2LITE_QK_ROPE_HEAD_DIM    64
#define DSV2LITE_QK_NOPE_HEAD_DIM    128
#define DSV2LITE_V_HEAD_DIM          128
#define DSV2LITE_N_ROUTED_EXPERTS    64
#define DSV2LITE_N_SHARED_EXPERTS    2
#define DSV2LITE_N_EXPERTS_PER_TOK   6
#define DSV2LITE_MOE_INTER_SIZE      1408
#define DSV2LITE_DENSE_INTER_SIZE    10944
#define DSV2LITE_FIRST_K_DENSE       1
#define DSV2LITE_VOCAB_SIZE          102400
#define DSV2LITE_MAX_SEQ_LEN         4096   /* pre-YaRN cap */
#define DSV2LITE_RMS_EPS             1e-6f
/* YaRN RoPE (rope_scaling in config.json). mscale==mscale_all_dim => _mscale=1. */
#define DSV2LITE_ROPE_THETA          10000.0f
#define DSV2LITE_ROPE_FACTOR         40.0f
#define DSV2LITE_ROPE_BETA_FAST      32.0f
#define DSV2LITE_ROPE_BETA_SLOW      1.0f
#define DSV2LITE_ROPE_ORIG_MAX       4096.0f

/* GLM-4.7-Flash (31.2B / 3.6B active) — secondary target */
#define GLM47_N_LAYERS               47
#define GLM47_HIDDEN                 2048
#define GLM47_N_HEADS                20
#define GLM47_N_KV_HEADS             20
#define GLM47_KV_LORA_RANK           512
#define GLM47_Q_LORA_RANK            768    /* GLM uses a query LoRA */
#define GLM47_QK_ROPE_HEAD_DIM       64
#define GLM47_QK_NOPE_HEAD_DIM       192
#define GLM47_V_HEAD_DIM             256
#define GLM47_N_ROUTED_EXPERTS       64
#define GLM47_N_SHARED_EXPERTS       1
#define GLM47_N_EXPERTS_PER_TOK      4
#define GLM47_MOE_INTER_SIZE         1536
#define GLM47_DENSE_INTER_SIZE       10240
#define GLM47_FIRST_K_DENSE          1
#define GLM47_VOCAB_SIZE             154880
#define GLM47_MAX_SEQ_LEN            4096   /* actual 202752; capped */
