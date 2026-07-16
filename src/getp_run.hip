/* ============================================================================
 * THIS IS THE ONLY FILE YOU MAY MODIFY.
 *
 * Your task: maximize end-to-end throughput (tok/s) of inference() over the
 * request batch, WITHOUT breaking correctness (`make eval` must still pass its
 * thresholds). The intended path is to port the compute to the GPU: start from
 * this correct CPU reference, then replace the forward_* calls with your own
 * kernels — allocate/upload in warm_up(), free in finish().
 *
 * The frozen harness (src/getp_eval.c) times inference() and prints your score.
 * The frozen reference kernels are declared in include/engine.h.
 * ==========================================================================*/
#include <stdio.h>
#include <stdlib.h>

#include "getp.h"       /* Requests + the warm_up/finish/inference contract */
#include "engine.h"     /* forward_unabsorbed / forward_absorbed / sample */
#include "tokenizer.h"  /* tokenizer_encode / tokenizer_eos_id */

void warm_up(Transformer *t) {
    /* Do not run inference here. Do one-time setup:
     *   - allocate scratch / device buffers
     *   - upload weights to the GPU
     * TODO(candidate). */
    (void)t;
}

void finish(Transformer *t) {
    /* One-time teardown mirroring warm_up(). TODO(candidate). */
    (void)t;
}

/* Reference implementation: greedy generate each request independently.
 * Per request: prefill the whole prompt (unabsorbed), then absorbed decode
 * until EOS or max_steps. Returns the total number of tokens generated. */
long long inference(Transformer *t, Requests *reqs) {
    const int   vocab       = t->config.vocab_size;
    const int   max_seq_len = t->config.max_seq_len;
    const int   eos         = t->tokenizer ? tokenizer_eos_id(t->tokenizer) : -1;

    if (!t->tokenizer) {
        fprintf(stderr, "inference: model_dir has no tokenizer.json\n");
        exit(EXIT_FAILURE);
    }

    int *prompt = (int *)malloc((size_t)max_seq_len * sizeof(int));
    long long total = 0;

    for (int r = 0; r < reqs->num_reqs; r++) {
        /* add_bos=0 matches the oracle/reference tokenization path. */
        int n_prompt = tokenizer_encode(t->tokenizer, reqs->prompts[r], 0,
                                        prompt, max_seq_len);
        if (n_prompt < 1) continue;

        /* Cap generation so prompt + decode stays within the KV cache. */
        int budget = max_seq_len - n_prompt;
        int steps  = reqs->max_steps < budget ? reqs->max_steps : budget;
        if (steps < 0) steps = 0;

        float *logits = forward_unabsorbed(t, prompt, n_prompt, NULL, NULL);
        int    tok    = sample(logits, vocab);
        int    pos    = n_prompt;
        int    n_out  = 0;

        while (n_out < steps && tok != eos) {
            reqs->out_tokens[r][n_out++] = tok;
            logits = forward_absorbed(t, tok, pos);
            tok    = sample(logits, vocab);
            pos++;
        }
        reqs->out_lens[r] = n_out;
        total += n_out;
    }

    free(prompt);
    return total;
}
