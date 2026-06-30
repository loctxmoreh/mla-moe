/* Standalone tokenizer harness for cross-checking against HuggingFace.
 *   tok_cli <model_dir> <text>
 * Prints a line of space-separated ids (encode, add_bos=0), then the decoded
 * round-trip text. Links only tokenizer.c + cJSON — no model weights loaded. */
#include <stdio.h>
#include <stdlib.h>
#include "tokenizer.h"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model_dir> <text>\n", argv[0]); return 2; }
    Tokenizer *t = tokenizer_load(argv[1]);
    if (!t) { fprintf(stderr, "tokenizer_load failed for %s\n", argv[1]); return 1; }
    int ids[8192];
    int n = tokenizer_encode(t, argv[2], 0, ids, 8192);
    for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? " " : "");
    printf("\n");
    char *s = tokenizer_decode(t, ids, n);
    printf("%s\n", s);
    free(s);
    tokenizer_free(t);
    return 0;
}
