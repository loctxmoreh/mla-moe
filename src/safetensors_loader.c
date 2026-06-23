/* Safetensors loader: vendor/safetensors.h parses headers; mmap, index.json
 * shard routing, and the TensorStore API live here. */
#define SAFETENSORS_IMPLEMENTATION  /* compile the vendor impl once */
#include "../vendor/safetensors.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "tensor.h"
#include "safetensors_loader.h"

struct TensorStore {
    void   **mmap_bases;        /* one per unique shard */
    size_t  *mmap_lens;
    int      n_shards;
    Tensor  *tensors;           /* flat array */
    size_t   n_tensors;
    safetensors_File *sf_files; /* kept alive: tensor names point into the mmap */
};

static void die(const char *msg) {
    fprintf(stderr, "safetensors_loader: %s\n", msg);
    exit(1);
}

/* Own impl: strndup isn't declared under -std=c11 -Wpedantic. */
static char *st_strndup(const char *p, size_t n) {
    char *s = (char*)malloc(n + 1);
    if (!s) die("out of memory (strndup)");
    memcpy(s, p, n);
    s[n] = '\0';
    return s;
}

/* Read an entire (small) file into a malloc'd, null-terminated buffer. */
static char *read_whole_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "safetensors_loader: cannot open %s\n", path);
        exit(1);
    }
    if (fseek(f, 0, SEEK_END) != 0) die("fseek failed");
    long sz = ftell(f);
    if (sz < 0) die("ftell failed");
    rewind(f);
    char *buf = (char*)malloc((size_t)sz + 1);
    if (!buf) die("out of memory (read_whole_file)");
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) die("short read on index file");
    buf[sz] = '\0';
    fclose(f);
    if (out_len) *out_len = (size_t)sz;
    return buf;
}

/* Extract unique shard filenames from the flat "weight_map" object
 * ({ "tensor.name": "shard.safetensors", ... }) with a simple string scan. */
static void add_unique_shard(char ***shards, int *n, int *cap,
                             const char *s, size_t len) {
    for (int i = 0; i < *n; i++)
        if (strlen((*shards)[i]) == len && memcmp((*shards)[i], s, len) == 0)
            return; /* already present */
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 8;
        *shards = (char**)realloc(*shards, (size_t)*cap * sizeof(char*));
        if (!*shards) die("out of memory (shard list)");
    }
    (*shards)[(*n)++] = st_strndup(s, len);
}

static char **parse_weight_map_shards(const char *json, int *n_shards_out) {
    const char *p = strstr(json, "\"weight_map\"");
    if (!p) die("index.json: no \"weight_map\" object found");
    p = strchr(p, '{');
    if (!p) die("index.json: malformed weight_map (no '{')");
    p++; /* step inside the object */

    char **shards = NULL;
    int n = 0, cap = 0;
    int depth = 1;

    while (*p && depth > 0) {
        if (*p == '}') { depth--; p++; continue; }
        if (*p == '{') { depth++; p++; continue; }
        if (*p == '"') {
            /* key string */
            p++;
            while (*p && *p != '"') p++;
            if (!*p) break;
            p++;                                /* past closing quote of key */
            while (*p && *p != ':') p++;        /* to ':' */
            if (*p == ':') p++;
            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
            /* value string = shard filename */
            if (*p == '"') {
                const char *vs = ++p;
                while (*p && *p != '"') p++;
                add_unique_shard(&shards, &n, &cap, vs, (size_t)(p - vs));
                if (*p) p++;                    /* past closing quote of value */
            }
            continue;
        }
        p++;
    }

    if (n == 0) die("index.json: weight_map contained no shard entries");
    *n_shards_out = n;
    return shards;
}

static void append_tensor(TensorStore *s, size_t *cap, Tensor t) {
    if (s->n_tensors == *cap) {
        *cap = *cap ? *cap * 2 : 1024;
        s->tensors = (Tensor*)realloc(s->tensors, *cap * sizeof(Tensor));
        if (!s->tensors) die("out of memory (tensor array)");
    }
    s->tensors[s->n_tensors++] = t;
}

TensorStore *st_load_sharded(const char *index_path, const char *shard_dir) {
    size_t index_len = 0;
    char *index_json = read_whole_file(index_path, &index_len);
    int n_shards = 0;
    char **shard_names = parse_weight_map_shards(index_json, &n_shards);
    free(index_json);

    TensorStore *store = (TensorStore*)calloc(1, sizeof(TensorStore));
    if (!store) die("out of memory (TensorStore)");
    store->n_shards    = n_shards;
    store->mmap_bases  = (void**) malloc((size_t)n_shards * sizeof(void*));
    store->mmap_lens   = (size_t*)malloc((size_t)n_shards * sizeof(size_t));
    store->sf_files    = (safetensors_File*)calloc((size_t)n_shards,
                                                    sizeof(safetensors_File));
    if (!store->mmap_bases || !store->mmap_lens || !store->sf_files)
        die("out of memory (store arrays)");

    size_t tensor_cap = 0;
    char path[4096];

    for (int i = 0; i < n_shards; i++) {
        snprintf(path, sizeof(path), "%s/%s", shard_dir, shard_names[i]);

        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "safetensors_loader: cannot open shard %s\n", path);
            exit(1);
        }
        struct stat sb;
        if (fstat(fd, &sb) != 0) die("fstat failed on shard");
        size_t len = (size_t)sb.st_size;

        void *base = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);                              /* mmap outlives the fd */
        if (base == MAP_FAILED) die("mmap failed on shard");

        store->mmap_bases[i] = base;
        store->mmap_lens[i]  = len;

        char *err = safetensors_file_init(base, (int64_t)len, &store->sf_files[i]);
        if (err) {
            fprintf(stderr, "safetensors_loader: parse error in %s: %s\n",
                    path, err);
            exit(1);
        }

        safetensors_File *sf = &store->sf_files[i];
        for (int j = 0; j < sf->num_tensors; j++) {
            safetensors_TensorDescriptor *d = &sf->tensors[j];

            /* Record dtype rather than asserting BF16: GLM-4.7-Flash has an F32
             * router bias per MoE layer. BF16-only consumers check Tensor.dtype. */
            if (d->n_dimensions > MAX_DIMS) {
                fprintf(stderr,
                        "safetensors_loader: tensor %.*s has %d dims (> MAX_DIMS=%d)\n",
                        d->name.len, d->name.ptr, d->n_dimensions, MAX_DIMS);
                exit(1);
            }

            Tensor t;
            t.name      = st_strndup(d->name.ptr, (size_t)d->name.len);
            t.data      = (bf16_t*)d->ptr;      /* direct pointer into mmap */
            t.dtype     = d->dtype;
            t.elem_size = (size_t)safetensors_dtype_size(d->dtype);
            t.ndim      = d->n_dimensions;
            t.n_elems = 1;
            for (int k = 0; k < d->n_dimensions; k++) {
                t.shape[k] = (size_t)d->shape[k];
                t.n_elems *= t.shape[k];
            }
            for (int k = d->n_dimensions; k < MAX_DIMS; k++)
                t.shape[k] = 0;

            append_tensor(store, &tensor_cap, t);
        }
    }

    for (int i = 0; i < n_shards; i++)
        free(shard_names[i]);
    free(shard_names);

    return store;
}

const Tensor *st_get(const TensorStore *store, const char *name) {
    if (!store || !name) return NULL;
    for (size_t i = 0; i < store->n_tensors; i++)
        if (strcmp(store->tensors[i].name, name) == 0)
            return &store->tensors[i];
    return NULL;
}

const Tensor *st_next(const TensorStore *store, size_t *iter) {
    if (!store || !iter) return NULL;
    if (*iter >= store->n_tensors) return NULL;
    return &store->tensors[(*iter)++];
}

size_t st_count(const TensorStore *store) {
    return store ? store->n_tensors : 0;
}

const char *st_dtype_name(int dtype) {
    return safetensors_dtype_name(dtype);
}

void st_free(TensorStore *store) {
    if (!store) return;

    /* free strdup'd names */
    for (size_t i = 0; i < store->n_tensors; i++)
        free((char*)store->tensors[i].name);
    free(store->tensors);

    /* free safetensors_File heap arrays + unmap shards */
    for (int i = 0; i < store->n_shards; i++) {
        free(store->sf_files[i].tensors);
        free(store->sf_files[i].metadata);
        munmap(store->mmap_bases[i], store->mmap_lens[i]);
    }
    free(store->sf_files);
    free(store->mmap_bases);
    free(store->mmap_lens);
    free(store);
}
