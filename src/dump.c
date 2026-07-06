/* Oracle-validation dump sink. See include/dump.h. Split out of run.c so the
 * forward passes carry the dump *calls* but not the file I/O or the layer-0
 * reorder bookkeeping. Always compiled; the real body is behind MLA_ENABLE_DUMP. */
#include "dump.h"

#ifdef MLA_ENABLE_DUMP
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *g_dump = NULL;
void run_set_dump(const char *dir) { g_dump = dir; }
int  mla_dumping(void) { return g_dump != NULL; }

void dump_bin(const char *name, const void *p, size_t n, size_t elem) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", g_dump, name);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "dump: cannot open %s\n", path); return; }
    fwrite(p, elem, n, f);
    fclose(f);
}

void dump_prefill_layer0(const float *qall, const float *kv_l, const float *knope,
                         const float *value, const float *scr, const float *wgt,
                         int NH, int n_prompt, int QKN, int QKR, int KVL, int VHD) {
    const int QHD = QKN + QKR;      /* qall per-head stride */
    const int KVD = KVL + QKR;      /* kv-cache per-position stride */
    /* q_nope/q_pe are [n_heads, seq, dim] (heads-first, like k_nope). */
    float *t1 = malloc((size_t)NH * n_prompt * QKN * sizeof(float));
    for (int h = 0; h < NH; h++)
        for (int p = 0; p < n_prompt; p++)
            memcpy(&t1[((size_t)h * n_prompt + p) * QKN],
                   &qall[((size_t)p * NH + h) * QHD], QKN * sizeof(float));
    dump_bin("prefill_layer0_q_nope", t1, (size_t)NH * n_prompt * QKN, sizeof(float));
    float *t2 = malloc((size_t)NH * n_prompt * QKR * sizeof(float));
    for (int h = 0; h < NH; h++)
        for (int p = 0; p < n_prompt; p++)
            memcpy(&t2[((size_t)h * n_prompt + p) * QKR],
                   &qall[((size_t)p * NH + h) * QHD + QKN], QKR * sizeof(float));
    dump_bin("prefill_layer0_q_pe", t2, (size_t)NH * n_prompt * QKR, sizeof(float));
    float *t3 = malloc((size_t)n_prompt * KVL * sizeof(float));
    float *t4 = malloc((size_t)n_prompt * QKR * sizeof(float));
    for (int p = 0; p < n_prompt; p++) {    /* c_kv [seq,kv_lora], k_pe [seq,qk_rope] */
        memcpy(&t3[(size_t)p * KVL], &kv_l[(size_t)p * KVD], KVL * sizeof(float));
        memcpy(&t4[(size_t)p * QKR], &kv_l[(size_t)p * KVD + KVL], QKR * sizeof(float));
    }
    dump_bin("prefill_layer0_c_kv", t3, (size_t)n_prompt * KVL, sizeof(float));
    dump_bin("prefill_layer0_k_pe", t4, (size_t)n_prompt * QKR, sizeof(float));
    dump_bin("prefill_layer0_k_nope", knope, (size_t)NH * n_prompt * QKN, sizeof(float));
    dump_bin("prefill_layer0_value", value, (size_t)NH * n_prompt * VHD, sizeof(float));
    dump_bin("prefill_layer0_attn_scores", scr, (size_t)NH * n_prompt * n_prompt, sizeof(float));
    dump_bin("prefill_layer0_attn_weights", wgt, (size_t)NH * n_prompt * n_prompt, sizeof(float));
    free(t1); free(t2); free(t3); free(t4);
}

#else
void run_set_dump(const char *dir) { (void)dir; }
#endif
