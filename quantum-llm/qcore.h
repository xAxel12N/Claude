/* qcore.h - shared types for QuantumGPT, a tiny quantum-inspired language model in C.
 *
 * The hidden state of every character is a complex "wavefunction". Attention weights
 * are squared overlaps |<q|k>|^2, each block runs a small circuit of unitary 2x2 gates,
 * and the next character is chosen by the Born rule P = |amplitude|^2 / sum. */
#ifndef QCORE_H
#define QCORE_H

#include <complex.h>
#include <stddef.h>
#include <stdint.h>

typedef double complex cpx;

/* ---------- model ---------- */
typedef struct {
    int V; /* vocabulary size (distinct bytes in the training text) */
    int D; /* complex width of the hidden state (even, divisible by H) */
    int H; /* attention heads */
    int L; /* number of blocks */
    int T; /* context length in characters */
    int R; /* depth of the unitary gate circuit in each block */
} Config;

typedef struct {
    size_t wq, wk, wv, wo; /* D*D complex each */
    size_t theta, phi;     /* 2R*(D/2) real gate angles */
    size_t psi;            /* D real phases */
    size_t bias;           /* D real modReLU thresholds */
    size_t wf;             /* D*D complex */
} BlockOff;

typedef struct {
    Config c;
    size_t emb, wout, n;  /* offsets in doubles, and total number of doubles */
    BlockOff *blk;
    double *p;            /* all parameters; complex ones are stored as (re, im) pairs */
    unsigned char *decay; /* 1 where weight decay applies (the matrices) */
} Model;

typedef struct Work Work; /* per-thread activations + gradient buffer */

int model_init(Model *m, Config c, uint64_t *rng);
void model_free(Model *m);
Work *work_new(const Model *m);
void work_free(Work *w);
double *work_grad(Work *w);
/* Runs the model over n tokens. With targets, returns the mean loss over the n positions. */
double model_forward(const Model *m, Work *w, const int *tok, const int *tgt, int n);
/* Adds scale * d(mean loss)/d(params) into the work's gradient buffer. Call after model_forward. */
void model_backward(const Model *m, Work *w, const int *tok, const int *tgt, int n, double scale);
int model_sample(const Model *m, Work *w, const int *ctx, int n, double temp, uint64_t *rng);

/* ---------- optimisation ---------- */
enum { OPT_SGD, OPT_MOMENTUM, OPT_ADAM, OPT_ADAMW };
enum { SCHED_CONST, SCHED_COSINE, SCHED_STEP };

typedef struct {
    int type, sched;
    double lr, min_lr, beta1, beta2, wd, clip;
    double anneal; /* "quantum annealing": gradient-noise temperature, cools with the lr */
    long warmup, steps;
} OptCfg;

typedef struct {
    double *m, *v;
    long t;
} OptState;

int opt_init(OptState *s, size_t n);
void opt_free(OptState *s);
double opt_lr(const OptCfg *o, long step);
double opt_clip(double *g, size_t n, double clip);
void opt_step(const OptCfg *o, OptState *s, const Model *m, const double *g, double lr, uint64_t *rng);
int opt_parse(const char *name);
int sched_parse(const char *name);
const char *opt_name(int type);
const char *sched_name(int sched);

/* ---------- data & checkpoints ---------- */
typedef struct {
    int V;
    unsigned char itos[256];
    int stoi[256];
    int *tok;   /* whole text, encoded */
    long n;
    long split; /* tokens [0, split) train, [split, n) validation */
} Data;

int data_load(Data *d, const char *path, int T);
void data_free(Data *d);
void data_batch(const Data *d, int val, int T, int *x, int *y, uint64_t *rng);
int data_encode(const Data *d, const char *s, int *out, int max);

int ckpt_save(const char *path, const Model *m, const OptState *s, const Data *d, long step, double best);
int ckpt_load(const char *path, Model *m, OptState *s, Data *d, long *step, double *best);

/* ---------- random numbers ---------- */
static inline uint64_t rng_u64(uint64_t *s)
{
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    return *s * 2685821657736338717ULL;
}
static inline double rng_unif(uint64_t *s) { return (rng_u64(s) >> 11) * (1.0 / 9007199254740992.0); }
double rng_normal(uint64_t *s);

#endif
