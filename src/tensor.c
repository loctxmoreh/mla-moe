/* Tensor debug dumps. bf16<->f32 casts are inline in tensor.h.
 * Output format ("f32:"/"hex:" lines) is parsed by check_tensors.py. */
#include <stdio.h>
#include "tensor.h"

void tensor_print_f32(const Tensor *t, size_t n) {
    if (n > t->n_elems) n = t->n_elems;
    printf("f32:");
    for (size_t i = 0; i < n; i++)
        printf(" %.6f", bf16_to_f32(t->data[i]));
    printf("\n");
}

void tensor_print_hex(const Tensor *t, size_t n) {
    if (n > t->n_elems) n = t->n_elems;
    printf("hex:");
    for (size_t i = 0; i < n; i++)
        printf(" %04x", (unsigned)t->data[i]);
    printf("\n");
}
