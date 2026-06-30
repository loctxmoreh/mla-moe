/* Byte-level BPE tokenizer for DeepSeek-V2-Lite — see tokenizer.h.
 *
 * Pipeline (mirrors tokenizer.json, reduced to ASCII):
 *   pre-tokenize -> byte-level encode each piece -> greedy BPE merge -> ids.
 * The pre-tokenizer is DeepSeek's Split/Digits/ByteLevel Sequence with the
 * letter/punct classes narrowed to ASCII and the CJK split dropped. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"
#include "cJSON.h"

/* ---- small string->int hash map (open addressing, linear probe) ---------- */

typedef struct { char *key; int val; } MapEntry;
typedef struct { MapEntry *e; size_t cap, n; } Map;

static size_t hash_bytes(const char *s, size_t len) {
    size_t h = 1469598103934665603ULL;            /* FNV-1a */
    for (size_t i = 0; i < len; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}

static void map_init(Map *m, size_t want) {
    m->cap = 16;
    while (m->cap < want * 2) m->cap <<= 1;        /* keep load factor < 0.5 */
    m->e = calloc(m->cap, sizeof(MapEntry));
    m->n = 0;
}

static void map_put(Map *m, const char *key, size_t len, int val) {
    size_t i = hash_bytes(key, len) & (m->cap - 1);
    while (m->e[i].key) {
        if (strlen(m->e[i].key) == len && memcmp(m->e[i].key, key, len) == 0) {
            m->e[i].val = val; return;             /* overwrite duplicate key */
        }
        i = (i + 1) & (m->cap - 1);
    }
    m->e[i].key = malloc(len + 1);
    memcpy(m->e[i].key, key, len); m->e[i].key[len] = '\0';
    m->e[i].val = val;
    m->n++;
}

/* returns val, or `dflt` if key absent */
static int map_get(const Map *m, const char *key, size_t len, int dflt) {
    size_t i = hash_bytes(key, len) & (m->cap - 1);
    while (m->e[i].key) {
        if (strlen(m->e[i].key) == len && memcmp(m->e[i].key, key, len) == 0)
            return m->e[i].val;
        i = (i + 1) & (m->cap - 1);
    }
    return dflt;
}

static void map_free(Map *m) {
    for (size_t i = 0; i < m->cap; i++) free(m->e[i].key);
    free(m->e);
}

/* ---- tokenizer state ----------------------------------------------------- */

struct Tokenizer {
    Map vocab;          /* byte-level token string -> id          */
    Map ranks;          /* "left right" merge string -> rank      */
    char **id2tok;      /* id -> byte-level token string (decode) */
    char *is_special;   /* id -> 1 if added/special token         */
    int n_ids;          /* length of id2tok / is_special          */
    int bos_id, eos_id;
    int byte2cp[256];   /* GPT-2 byte -> byte-level codepoint      */
    int cp2byte[512];   /* inverse (codepoint <= 323)              */
    char *bytetok[256]; /* byte -> its byte-level UTF-8 string     */
};

/* ---- file + json helpers ------------------------------------------------- */

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return NULL; }
    buf[n] = '\0'; fclose(f);
    return buf;
}

static int json_int(const cJSON *o, const char *k, int dflt) {
    const cJSON *v = cJSON_GetObjectItemCaseSensitive(o, k);
    return cJSON_IsNumber(v) ? (int)v->valuedouble : dflt;
}

/* ---- GPT-2 byte-level alphabet ------------------------------------------- */

static int bytelevel_printable(int b) {
    return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
}

/* utf-8 encode one codepoint (cp < 2048 here) into dst; returns byte count */
static int utf8_encode(int cp, char *dst) {
    if (cp < 0x80) { dst[0] = (char)cp; return 1; }
    dst[0] = (char)(0xC0 | (cp >> 6));
    dst[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
}

/* decode one utf-8 codepoint from s (<=3 bytes); advances *i */
static int utf8_decode(const char *s, size_t *i) {
    unsigned char c = (unsigned char)s[*i];
    if (c < 0x80) { (*i)++; return c; }
    if ((c & 0xE0) == 0xC0) {
        int cp = ((c & 0x1F) << 6) | ((unsigned char)s[*i + 1] & 0x3F);
        *i += 2; return cp;
    }
    int cp = ((c & 0x0F) << 12) | (((unsigned char)s[*i + 1] & 0x3F) << 6)
           | ((unsigned char)s[*i + 2] & 0x3F);
    *i += 3; return cp;
}

static void build_bytelevel(Tokenizer *t) {
    int next = 256;
    for (int b = 0; b < 256; b++)
        t->byte2cp[b] = bytelevel_printable(b) ? b : next++;
    for (int i = 0; i < 512; i++) t->cp2byte[i] = -1;
    for (int b = 0; b < 256; b++) {
        t->cp2byte[t->byte2cp[b]] = b;
        char buf[4]; int n = utf8_encode(t->byte2cp[b], buf);
        t->bytetok[b] = malloc(n + 1);
        memcpy(t->bytetok[b], buf, n); t->bytetok[b][n] = '\0';
    }
}

/* ---- load ---------------------------------------------------------------- */

Tokenizer *tokenizer_load(const char *model_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/tokenizer.json", model_dir);
    char *text = read_file(path);
    if (!text) return NULL;
    cJSON *root = cJSON_Parse(text);
    free(text);
    if (!root) return NULL;

    const cJSON *model = cJSON_GetObjectItemCaseSensitive(root, "model");
    const cJSON *vocab = cJSON_GetObjectItemCaseSensitive(model, "vocab");
    const cJSON *merges = cJSON_GetObjectItemCaseSensitive(model, "merges");
    const cJSON *added = cJSON_GetObjectItemCaseSensitive(root, "added_tokens");
    if (!cJSON_IsObject(vocab) || !cJSON_IsArray(merges)) { cJSON_Delete(root); return NULL; }

    Tokenizer *t = calloc(1, sizeof(Tokenizer));
    t->bos_id = t->eos_id = -1;
    build_bytelevel(t);

    /* pass 1: find max id to size id2tok */
    int max_id = 0;
    for (const cJSON *it = vocab->child; it; it = it->next)
        if ((int)it->valuedouble > max_id) max_id = (int)it->valuedouble;
    for (const cJSON *it = added ? added->child : NULL; it; it = it->next) {
        int id = json_int(it, "id", -1);
        if (id > max_id) max_id = id;
    }
    t->n_ids = max_id + 1;
    t->id2tok = calloc(t->n_ids, sizeof(char *));
    t->is_special = calloc(t->n_ids, 1);

    /* vocab: string -> id, and id -> string */
    map_init(&t->vocab, (size_t)cJSON_GetArraySize(vocab));
    for (const cJSON *it = vocab->child; it; it = it->next) {
        int id = (int)it->valuedouble;
        size_t klen = strlen(it->string);
        map_put(&t->vocab, it->string, klen, id);
        if (!t->id2tok[id]) {
            t->id2tok[id] = malloc(klen + 1);
            memcpy(t->id2tok[id], it->string, klen + 1);
        }
    }

    /* merges: "left right" -> rank (array index) */
    map_init(&t->ranks, (size_t)cJSON_GetArraySize(merges));
    int rank = 0;
    for (const cJSON *it = merges->child; it; it = it->next, rank++) {
        const char *s = cJSON_IsString(it) ? it->valuestring : NULL;
        if (s) map_put(&t->ranks, s, strlen(s), rank);
    }

    /* added/special tokens: id -> content, mark special */
    for (const cJSON *it = added ? added->child : NULL; it; it = it->next) {
        int id = json_int(it, "id", -1);
        const cJSON *c = cJSON_GetObjectItemCaseSensitive(it, "content");
        if (id < 0 || id >= t->n_ids || !cJSON_IsString(c)) continue;
        free(t->id2tok[id]);
        t->id2tok[id] = malloc(strlen(c->valuestring) + 1);
        strcpy(t->id2tok[id], c->valuestring);
        t->is_special[id] = 1;
    }
    cJSON_Delete(root);

    /* bos/eos from config.json (authoritative) */
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    char *cfgtext = read_file(path);
    if (cfgtext) {
        cJSON *cfg = cJSON_Parse(cfgtext);
        if (cfg) {
            t->bos_id = json_int(cfg, "bos_token_id", -1);
            t->eos_id = json_int(cfg, "eos_token_id", -1);
            cJSON_Delete(cfg);
        }
        free(cfgtext);
    }
    return t;
}

/* ---- ASCII pre-tokenizer ------------------------------------------------- */

typedef struct { const char *s; int len; } Piece;
typedef struct { Piece *p; int n, cap; } PieceList;

static void pl_push(PieceList *l, const char *s, int len) {
    if (len <= 0) return;
    if (l->n == l->cap) { l->cap = l->cap ? l->cap * 2 : 16; l->p = realloc(l->p, l->cap * sizeof(Piece)); }
    l->p[l->n].s = s; l->p[l->n].len = len; l->n++;
}

static int is_space(unsigned char c) { return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v'; }
static int is_alpha(unsigned char c) { return (c>='A'&&c<='Z')||(c>='a'&&c<='z'); }
static int is_digit(unsigned char c) { return c>='0'&&c<='9'; }
/* literal [!-/:-~]: 0x21-0x2F and 0x3A-0x7E */
static int is_punct(unsigned char c) { return (c>=0x21&&c<=0x2F)||(c>=0x3A&&c<=0x7E); }

/* Split [\r\n] (Isolated): each CR/LF becomes its own piece. */
static void split_newlines(const PieceList *in, PieceList *out) {
    for (int k = 0; k < in->n; k++) {
        const char *s = in->p[k].s; int len = in->p[k].len, start = 0;
        for (int i = 0; i < len; i++) {
            if (s[i]=='\r' || s[i]=='\n') {
                pl_push(out, s + start, i - start);
                pl_push(out, s + i, 1);
                start = i + 1;
            }
        }
        pl_push(out, s + start, len - start);
    }
}

/* Split \s?[class]+ (Isolated): optional one leading space + run of `cls`. */
static void split_ws_class(const PieceList *in, PieceList *out, int (*cls)(unsigned char)) {
    for (int k = 0; k < in->n; k++) {
        const char *s = in->p[k].s; int len = in->p[k].len, gap = 0, i = 0;
        while (i < len) {
            unsigned char c = (unsigned char)s[i];
            int mstart = -1, body = i;
            if (cls(c)) mstart = i;
            else if (is_space(c) && i + 1 < len && cls((unsigned char)s[i+1])) { mstart = i; body = i + 1; }
            if (mstart >= 0) {
                int j = body;
                while (j < len && cls((unsigned char)s[j])) j++;
                pl_push(out, s + gap, mstart - gap);   /* preceding gap */
                pl_push(out, s + mstart, j - mstart);   /* the match     */
                i = gap = j;
            } else i++;
        }
        pl_push(out, s + gap, len - gap);
    }
}

/* Split \s+$ (Isolated): isolate a maximal trailing whitespace run. */
static void split_trailing_ws(const PieceList *in, PieceList *out) {
    for (int k = 0; k < in->n; k++) {
        const char *s = in->p[k].s; int len = in->p[k].len, r = len;
        while (r > 0 && is_space((unsigned char)s[r-1])) r--;
        if (r > 0 && r < len) { pl_push(out, s, r); pl_push(out, s + r, len - r); }
        else pl_push(out, s, len);
    }
}

/* Digits (individual_digits): each ASCII digit its own piece. */
static void split_digits(const PieceList *in, PieceList *out) {
    for (int k = 0; k < in->n; k++) {
        const char *s = in->p[k].s; int len = in->p[k].len, start = 0;
        for (int i = 0; i < len; i++) {
            if (is_digit((unsigned char)s[i])) {
                pl_push(out, s + start, i - start);
                pl_push(out, s + i, 1);
                start = i + 1;
            }
        }
        pl_push(out, s + start, len - start);
    }
}

/* ---- BPE merge ----------------------------------------------------------- */

typedef struct { int off, len; } Sym;   /* span into the byte-level buffer */

/* Greedily merge symbols by lowest merge rank; map result to ids. */
static int bpe_piece(const Tokenizer *t, const char *enc, Sym *sym, int n,
                     int *out, int max, int nout) {
    char key[256];
    for (;;) {
        int best = -1, best_rank = 0;
        for (int i = 0; i + 1 < n; i++) {
            int l1 = sym[i].len, l2 = sym[i+1].len;
            if (l1 + 1 + l2 >= (int)sizeof(key)) continue;
            memcpy(key, enc + sym[i].off, l1);
            key[l1] = ' ';
            memcpy(key + l1 + 1, enc + sym[i+1].off, l2);
            int r = map_get(&t->ranks, key, l1 + 1 + l2, -1);
            if (r >= 0 && (best < 0 || r < best_rank)) { best = i; best_rank = r; }
        }
        if (best < 0) break;
        sym[best].len += sym[best+1].len;            /* spans are contiguous */
        memmove(&sym[best+1], &sym[best+2], (n - best - 2) * sizeof(Sym));
        n--;
    }
    for (int i = 0; i < n && nout < max; i++) {
        int id = map_get(&t->vocab, enc + sym[i].off, sym[i].len, -1);
        if (id >= 0) out[nout++] = id;
    }
    return nout;
}

/* ---- encode -------------------------------------------------------------- */

int tokenizer_encode(const Tokenizer *t, const char *text, int add_bos,
                     int *out_ids, int max_ids) {
    int n = 0;
    if (add_bos && t->bos_id >= 0 && n < max_ids) out_ids[n++] = t->bos_id;

    /* pre-tokenize: chained Split stages over piece spans of `text` */
    int tlen = (int)strlen(text);
    PieceList a = {0}, b = {0};
    pl_push(&a, text, tlen);
    split_newlines(&a, &b);                                   a.n = 0;
    split_ws_class(&b, &a, is_alpha);                         b.n = 0;
    split_ws_class(&a, &b, is_punct);                         a.n = 0;
    split_trailing_ws(&b, &a);                                b.n = 0;
    split_digits(&a, &b);                                     /* result in b */

    /* byte-level encode + BPE per piece */
    char *enc = NULL; Sym *sym = NULL; int enc_cap = 0, sym_cap = 0;
    for (int k = 0; k < b.n; k++) {
        int plen = b.p[k].len;
        if (plen * 2 + 1 > enc_cap) { enc_cap = plen * 2 + 1; enc = realloc(enc, enc_cap); }
        if (plen > sym_cap) { sym_cap = plen; sym = realloc(sym, sym_cap * sizeof(Sym)); }
        int eoff = 0;
        for (int i = 0; i < plen; i++) {
            unsigned char by = (unsigned char)b.p[k].s[i];
            int slen = (int)strlen(t->bytetok[by]);
            memcpy(enc + eoff, t->bytetok[by], slen);
            sym[i].off = eoff; sym[i].len = slen;
            eoff += slen;
        }
        n = bpe_piece(t, enc, sym, plen, out_ids, max_ids, n);
    }
    free(enc); free(sym); free(a.p); free(b.p);
    return n;
}

/* ---- decode -------------------------------------------------------------- */

char *tokenizer_decode(const Tokenizer *t, const int *ids, int n) {
    size_t cap = 64, len = 0;
    char *out = malloc(cap);
    for (int k = 0; k < n; k++) {
        int id = ids[k];
        if (id < 0 || id >= t->n_ids || !t->id2tok[id] || t->is_special[id]) continue;
        const char *s = t->id2tok[id];
        size_t i = 0, slen = strlen(s);
        while (i < slen) {
            int cp = utf8_decode(s, &i);
            int by = (cp >= 0 && cp < 512) ? t->cp2byte[cp] : -1;
            if (by < 0) continue;
            if (len + 1 >= cap) { cap *= 2; out = realloc(out, cap); }
            out[len++] = (char)by;
        }
    }
    out[len] = '\0';
    return out;
}

int tokenizer_bos_id(const Tokenizer *t) { return t->bos_id; }
int tokenizer_eos_id(const Tokenizer *t) { return t->eos_id; }

void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    map_free(&t->vocab);
    map_free(&t->ranks);
    for (int i = 0; i < t->n_ids; i++) free(t->id2tok[i]);
    free(t->id2tok); free(t->is_special);
    for (int i = 0; i < 256; i++) free(t->bytetok[i]);
    free(t);
}
