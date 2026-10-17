/* data.c - text loading, character vocabulary, batches and checkpoints. */
#include "qcore.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int data_load(Data *d, const char *path, int T)
{
    memset(d, 0, sizeof *d);
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(n > 0 ? n : 1);
    if (!buf || fread(buf, 1, n, f) != (size_t)n) {
        fprintf(stderr, "%s: read failed\n", path);
        fclose(f);
        free(buf);
        return -1;
    }
    fclose(f);
    if (n < 2L * (T + 2)) {
        fprintf(stderr, "%s: text too short (%ld bytes) for context %d\n", path, n, T);
        free(buf);
        return -1;
    }

    int seen[256] = {0};
    for (long i = 0; i < n; i++) seen[buf[i]] = 1;
    for (int c = 0; c < 256; c++) {
        d->stoi[c] = -1;
        if (seen[c]) d->itos[d->V] = (unsigned char)c, d->stoi[c] = d->V++;
    }
    d->tok = malloc(n * sizeof *d->tok);
    if (!d->tok) {
        free(buf);
        return -1;
    }
    for (long i = 0; i < n; i++) d->tok[i] = d->stoi[buf[i]];
    free(buf);
    d->n = n;
    d->split = n * 9 / 10;
    if (n - d->split < T + 2) d->split = n; /* too small to hold out: validate on train */
    return 0;
}

void data_free(Data *d)
{
    free(d->tok);
    d->tok = NULL;
}

void data_batch(const Data *d, int val, int T, int *x, int *y, uint64_t *rng)
{
    long lo = 0, hi = d->split;
    if (val && d->split < d->n) lo = d->split, hi = d->n;
    long start = lo + (long)(rng_u64(rng) % (uint64_t)(hi - lo - T - 1));
    for (int i = 0; i < T; i++) {
        x[i] = d->tok[start + i];
        y[i] = d->tok[start + i + 1];
    }
}

int data_encode(const Data *d, const char *s, int *out, int max)
{
    int n = 0;
    for (; *s && n < max; s++)
        if (d->stoi[(unsigned char)*s] >= 0) out[n++] = d->stoi[(unsigned char)*s];
    return n;
}

static const char MAGIC[8] = "QGPT\0\0\0\1";

int ckpt_save(const char *path, const Model *m, const OptState *s, const Data *d, long step, double best)
{
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) {
        perror(tmp);
        return -1;
    }
    int ok = fwrite(MAGIC, 8, 1, f) == 1 && fwrite(&m->c, sizeof m->c, 1, f) == 1 &&
             fwrite(d->itos, 1, m->c.V, f) == (size_t)m->c.V && fwrite(&step, sizeof step, 1, f) == 1 &&
             fwrite(&best, sizeof best, 1, f) == 1 && fwrite(m->p, sizeof *m->p, m->n, f) == m->n &&
             fwrite(&s->t, sizeof s->t, 1, f) == 1 && fwrite(s->m, sizeof *s->m, m->n, f) == m->n &&
             fwrite(s->v, sizeof *s->v, m->n, f) == m->n;
    if (fclose(f) || !ok || rename(tmp, path)) { /* write-then-rename: a crash never leaves half a file */
        fprintf(stderr, "%s: save failed\n", path);
        remove(tmp);
        return -1;
    }
    return 0;
}

int ckpt_load(const char *path, Model *m, OptState *s, Data *d, long *step, double *best)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return -1;
    }
    char magic[8];
    Config c;
    memset(d, 0, sizeof *d);
    uint64_t rng = 1;
    int ok = fread(magic, 8, 1, f) == 1 && !memcmp(magic, MAGIC, 8) && fread(&c, sizeof c, 1, f) == 1 &&
             c.V > 1 && c.V <= 256 && fread(d->itos, 1, c.V, f) == (size_t)c.V &&
             fread(step, sizeof *step, 1, f) == 1 && fread(best, sizeof *best, 1, f) == 1 &&
             model_init(m, c, &rng) == 0;
    if (ok) {
        ok = fread(m->p, sizeof *m->p, m->n, f) == m->n;
        if (ok && s)
            ok = opt_init(s, m->n) == 0 && fread(&s->t, sizeof s->t, 1, f) == 1 &&
                 fread(s->m, sizeof *s->m, m->n, f) == m->n && fread(s->v, sizeof *s->v, m->n, f) == m->n;
    }
    fclose(f);
    if (!ok) {
        fprintf(stderr, "%s: not a valid QuantumGPT checkpoint\n", path);
        return -1;
    }
    d->V = c.V;
    for (int i = 0; i < 256; i++) d->stoi[i] = -1;
    for (int i = 0; i < c.V; i++) d->stoi[d->itos[i]] = i;
    return 0;
}
