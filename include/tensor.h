/* include/tensor.h */
#pragma once
#include <stdint.h>
#include <stddef.h>

/* BF16 is stored as raw uint16_t — upper 16 bits of IEEE 754 float32 */
typedef uint16_t bf16_t;

static inline float bf16_to_f32(bf16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

static inline bf16_t f32_to_bf16(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return (bf16_t)(u >> 16);   /* truncate, no rounding — matches PyTorch default */
}

#define MAX_DIMS 4

/* Raw safetensors dtype codes (values match hsnyder/safetensors.h). */
enum {
    TENSOR_DTYPE_F32  = 1,
    TENSOR_DTYPE_BF16 = 3,
};

typedef struct {
    const char *name;       /* not owned */
    bf16_t     *data;       /* into mmap, not owned; valid as bf16_t* iff dtype==BF16 */
    int         dtype;      /* TENSOR_DTYPE_* */
    size_t      elem_size;
    int         ndim;
    size_t      shape[MAX_DIMS];
    size_t      n_elems;
} Tensor;

void tensor_print_f32(const Tensor *t, size_t n);  /* first n elems as float */
void tensor_print_hex(const Tensor *t, size_t n);  /* first n elems as u16 hex */
