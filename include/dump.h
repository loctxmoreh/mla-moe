/* Oracle-validation dumps. Compiled in only with -DMLA_ENABLE_DUMP (build with
 * `make DUMP=1`); by default DUMPING() is a compile-time 0, so every dump block
 * and its scratch is dead-code-eliminated and production builds carry no dump
 * code, branches, or I/O. Set the output dir at runtime via run_set_dump().
 * Implemented in src/dump.c. */
#ifndef MLA_DUMP_H
#define MLA_DUMP_H

#include <stddef.h>
#include <stdint.h>

/* Set the dump output directory (no-op in non-DUMP builds). */
void run_set_dump(const char *dir);

#ifdef MLA_ENABLE_DUMP
int  mla_dumping(void);
void dump_bin(const char *name, const void *p, size_t n, size_t elem);
/* Reorder + write the prefill layer-0 MLA internals in the oracle's dump orders
 * (q_nope/q_pe/c_kv/k_pe reordered; k_nope/value/scores/weights as-is). */
void dump_prefill_layer0(const float *qall, const float *kv_l, const float *knope,
                         const float *value, const float *scr, const float *wgt,
                         int NH, int n_prompt, int QKN, int QKR, int KVL, int VHD);
#define DUMPING()              mla_dumping()
#define DUMP_F32(name, ptr, n) dump_bin((name), (ptr), (n), sizeof(float))
#define DUMP_I32(name, ptr, n) dump_bin((name), (ptr), (n), sizeof(int32_t))
#else
#define DUMPING() 0
/* discard args (marks them used) so dead dump blocks don't warn */
#define DUMP_F32(name, ptr, n) ((void)(name), (void)(ptr), (void)(n))
#define DUMP_I32(name, ptr, n) ((void)(name), (void)(ptr), (void)(n))
#define dump_prefill_layer0(...) ((void)0)
#endif

#endif /* MLA_DUMP_H */
