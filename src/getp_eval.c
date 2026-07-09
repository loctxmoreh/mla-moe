/* DO NOT MODIFY THIS FILE.
 *
 * Frozen batch-throughput harness. Reads a fixed request set, times the
 * candidate's inference(), and reports one end-to-end number:
 *   achieved throughput TPS (tok/s)
 * warm_up() and finish() are timed separately and excluded from throughput.
 * Ported in spirit from last year's gpt-oss getp-csrc/getp_eval.cpp. */
#define _POSIX_C_SOURCE 200809L   /* clock_gettime + getline / strdup / ssize_t */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "getp.h"

#define GETP_DEFAULT_STEPS 128

/* Monotonic wall-clock in ms — the timing primitive for throughput. */
static double getp_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

/* Read `req_file`: line 0 is the request count, each following line is one
 * prompt (verbatim, trailing newline stripped). Allocates Requests buffers. */
static void build_requests(Requests *reqs, const char *req_file, int max_steps) {
    FILE *f = fopen(req_file, "r");
    if (!f) {
        fprintf(stderr, "getp: cannot open request file: %s\n", req_file);
        exit(EXIT_FAILURE);
    }

    char   *line = NULL;
    size_t  cap  = 0;
    ssize_t len;

    /* line 0: number of requests */
    len = getline(&line, &cap, f);
    if (len < 0) {
        fprintf(stderr, "getp: empty request file: %s\n", req_file);
        exit(EXIT_FAILURE);
    }
    int num_reqs = atoi(line);
    if (num_reqs <= 0) {
        fprintf(stderr, "getp: bad request count on line 1: '%s'\n", line);
        exit(EXIT_FAILURE);
    }

    reqs->num_reqs   = num_reqs;
    reqs->max_steps  = max_steps;
    reqs->prompts    = (char **)calloc((size_t)num_reqs, sizeof(char *));
    reqs->out_tokens = (int  **)calloc((size_t)num_reqs, sizeof(int *));
    reqs->out_lens   = (int   *)calloc((size_t)num_reqs, sizeof(int));

    int idx = 0;
    while (idx < num_reqs && (len = getline(&line, &cap, f)) >= 0) {
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        reqs->prompts[idx]    = strdup(line);
        reqs->out_tokens[idx] = (int *)calloc((size_t)max_steps, sizeof(int));
        idx++;
    }
    free(line);
    fclose(f);

    if (idx < num_reqs) {
        fprintf(stderr, "getp: expected %d prompts, found %d in %s\n",
                num_reqs, idx, req_file);
        exit(EXIT_FAILURE);
    }
    printf("Num requests: %d (max_steps=%d)\n", num_reqs, max_steps);
    fflush(stdout);
}

static void free_requests(Requests *reqs) {
    for (int i = 0; i < reqs->num_reqs; i++) {
        free(reqs->prompts[i]);
        free(reqs->out_tokens[i]);
    }
    free(reqs->prompts);
    free(reqs->out_tokens);
    free(reqs->out_lens);
}

/* One line per request: space-separated generated token ids (same shape as
 * tests/eval/<model>/completions.i32.txt, so the output can be re-scored). */
static void write_outputfile(const char *out_file, Requests *reqs) {
    FILE *f = fopen(out_file, "w");
    if (!f) {
        fprintf(stderr, "getp: cannot write output file: %s\n", out_file);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < reqs->num_reqs; i++) {
        for (int j = 0; j < reqs->out_lens[i]; j++)
            fprintf(f, "%s%d", j ? " " : "", reqs->out_tokens[i][j]);
        fputc('\n', f);
    }
    fclose(f);
}

void getp(Transformer *t, const char *req_file, const char *out_file, int steps) {
    if (!req_file || !out_file) {
        fprintf(stderr, "getp: need <requests_file> and <output_file>\n");
        exit(EXIT_FAILURE);
    }
    if (steps <= 0) steps = GETP_DEFAULT_STEPS;

    Requests reqs;
    build_requests(&reqs, req_file, steps);

    double start, end;

    start = getp_now_ms();
    warm_up(t);
    end = getp_now_ms();
    printf("\nwarm up elapsed time(s): %f\n", (end - start) / 1e3);
    fflush(stdout);

    start = getp_now_ms();
    long long num_gen_tokens = inference(t, &reqs);
    end = getp_now_ms();
    /* Your goal is to achieve best throughput (= reduce elapsed time)! */
    printf("\nelapsed time(s): %f, achieved throughput TPS (tok/s): %f\n",
           (end - start) / 1e3,
           (double)num_gen_tokens / (end - start) * 1e3);
    fflush(stdout);

    write_outputfile(out_file, &reqs);

    start = getp_now_ms();
    finish(t);
    end = getp_now_ms();
    printf("\nfinish elapsed time(s): %f\n", (end - start) / 1e3);
    fflush(stdout);

    free_requests(&reqs);
}
