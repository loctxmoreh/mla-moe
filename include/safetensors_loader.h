/* include/safetensors_loader.h */
#ifndef SAFETENSORS_LOADER_H
#define SAFETENSORS_LOADER_H
#include "tensor.h"

/* Holds all mmapped shards + a flat Tensor[]; Tensor.data points into the
 * mmaps (zero copy). vendor/safetensors.h parses each shard header; this
 * wrapper does mmap, index.json shard routing, and name strdup. */
typedef struct TensorStore TensorStore;

/* Load a sharded model. index_path: model.safetensors.index.json; shard_dir:
 * directory holding the shards. */
TensorStore *st_load_sharded(const char *index_path, const char *shard_dir);

/* Look up a tensor by exact name. Returns NULL if not found. */
const Tensor *st_get(const TensorStore *store, const char *name);

/* Iterate all tensors. Set *iter = 0 before first call. */
const Tensor *st_next(const TensorStore *store, size_t *iter);

/* Total number of tensors */
size_t st_count(const TensorStore *store);

/* Human-readable name for a raw safetensors dtype code (e.g. "BF16", "F32"). */
const char *st_dtype_name(int dtype);

void st_free(TensorStore *store);

#endif /* SAFETENSORS_LOADER_H */
