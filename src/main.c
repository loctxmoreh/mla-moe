/* Weight inspection CLI:
 *   mla-moe list    <index_json> <shard_dir>
 *   mla-moe inspect <index_json> <shard_dir> <tensor_name> [n_elements]
 *   mla-moe dump    <index_json> <shard_dir> <tensor_name> [n] [outfile] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"
#include "safetensors_loader.h"

static void print_shape(const Tensor *t) {
    for (int i = 0; i < t->ndim; i++)
        printf("%s%zu", i ? "," : "", t->shape[i]);
}

static int cmd_list(const char *index_path, const char *shard_dir) {
    TensorStore *store = st_load_sharded(index_path, shard_dir);
    size_t iter = 0;
    const Tensor *t;
    while ((t = st_next(store, &iter)) != NULL) {
        printf("%s\tshape=", t->name);
        print_shape(t);
        printf("\tn=%zu\tdtype=%s\n", t->n_elems, st_dtype_name(t->dtype));
    }
    printf("# %zu tensors\n", st_count(store));
    st_free(store);
    return 0;
}

static int cmd_inspect(const char *index_path, const char *shard_dir,
                       const char *name, size_t n) {
    TensorStore *store = st_load_sharded(index_path, shard_dir);
    const Tensor *t = st_get(store, name);
    if (!t) {
        fprintf(stderr, "inspect: tensor not found: %s\n", name);
        st_free(store);
        return 1;
    }
    printf("%s  shape=", t->name);
    print_shape(t);
    printf("  n=%zu  dtype=%s\n", t->n_elems, st_dtype_name(t->dtype));
    if (t->dtype != TENSOR_DTYPE_BF16) {   /* hex/f32 dump assumes BF16 */
        fprintf(stderr, "inspect: %s is %s, not BF16 — hex/f32 dump skipped\n",
                t->name, st_dtype_name(t->dtype));
        st_free(store);
        return 1;
    }
    tensor_print_hex(t, n);
    tensor_print_f32(t, n);
    st_free(store);
    return 0;
}

/* Dump first `n` raw u16 words, one %04x per line, to outfile (or stdout if
 * NULL/"-"). Plain format for diffing against the Python dumper. */
static int cmd_dump(const char *index_path, const char *shard_dir,
                    const char *name, size_t n, const char *outfile) {
    TensorStore *store = st_load_sharded(index_path, shard_dir);
    const Tensor *t = st_get(store, name);
    if (!t) {
        fprintf(stderr, "dump: tensor not found: %s\n", name);
        st_free(store);
        return 1;
    }
    if (n > t->n_elems) n = t->n_elems;

    FILE *out = stdout;
    if (outfile && strcmp(outfile, "-") != 0) {
        out = fopen(outfile, "w");
        if (!out) {
            fprintf(stderr, "dump: cannot open %s for writing\n", outfile);
            st_free(store);
            return 1;
        }
    }
    for (size_t i = 0; i < n; i++)
        fprintf(out, "%04x\n", (unsigned)t->data[i]);
    if (out != stdout) fclose(out);
    st_free(store);
    return 0;
}

static int usage(const char *argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s list    <index_json> <shard_dir>\n"
        "  %s inspect <index_json> <shard_dir> <tensor_name> [n_elements]\n"
        "  %s dump    <index_json> <shard_dir> <tensor_name> [n] [outfile]\n",
        argv0, argv0, argv0);
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc < 2) return usage(argv[0]);

    if (strcmp(argv[1], "list") == 0) {
        if (argc < 4) return usage(argv[0]);
        return cmd_list(argv[2], argv[3]);
    }
    if (strcmp(argv[1], "inspect") == 0) {
        if (argc < 5) return usage(argv[0]);
        size_t n = (argc > 5) ? (size_t)strtoul(argv[5], NULL, 10) : 16;
        return cmd_inspect(argv[2], argv[3], argv[4], n);
    }
    if (strcmp(argv[1], "dump") == 0) {
        if (argc < 5) return usage(argv[0]);
        size_t n = (argc > 5) ? (size_t)strtoul(argv[5], NULL, 10) : 64;
        const char *outfile = (argc > 6) ? argv[6] : NULL;
        return cmd_dump(argv[2], argv[3], argv[4], n, outfile);
    }
    return usage(argv[0]);
}
