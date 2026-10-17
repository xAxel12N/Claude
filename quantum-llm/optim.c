/* optim.c - learning techniques: optimisers, learning-rate schedules, gradient
 * clipping, weight decay and "quantum annealing" gradient noise. */
#include "qcore.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

double rng_normal(uint64_t *s)
{
    double u1 = rng_unif(s), u2 = rng_unif(s);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static const char *OPT_NAMES[] = {"sgd", "momentum", "adam", "adamw"};
static const char *SCHED_NAMES[] = {"const", "cosine", "step"};

const char *opt_name(int type) { return OPT_NAMES[type]; }
const char *sched_name(int sched) { return SCHED_NAMES[sched]; }

int opt_parse(const char *name)
{
    for (int i = 0; i < 4; i++)
        if (!strcmp(name, OPT_NAMES[i])) return i;
    return -1;
}

int sched_parse(const char *name)
{
    for (int i = 0; i < 3; i++)
        if (!strcmp(name, SCHED_NAMES[i])) return i;
    return -1;
}

int opt_init(OptState *s, size_t n)
{
    s->m = calloc(n, sizeof *s->m);
    s->v = calloc(n, sizeof *s->v);
    s->t = 0;
    return s->m && s->v ? 0 : -1;
}

void opt_free(OptState *s)
{
    free(s->m);
    free(s->v);
    memset(s, 0, sizeof *s);
}

/* Linear warmup, then constant / cosine decay to min_lr / halve every quarter of the run. */
double opt_lr(const OptCfg *o, long step)
{
    if (step < o->warmup) return o->lr * (step + 1) / o->warmup;
    double span = o->steps - o->warmup, prog = span > 0 ? (step - o->warmup) / span : 1;
    if (prog > 1) prog = 1;
    switch (o->sched) {
    case SCHED_COSINE: return o->min_lr + (o->lr - o->min_lr) * 0.5 * (1 + cos(M_PI * prog));
    case SCHED_STEP: return fmax(o->min_lr, o->lr * pow(0.5, floor(prog * 4)));
    default: return o->lr;
    }
}

/* Rescales g so its global norm is at most clip (if clip > 0). Returns the norm before clipping. */
double opt_clip(double *g, size_t n, double clip)
{
    double s = 0;
    for (size_t i = 0; i < n; i++) s += g[i] * g[i];
    s = sqrt(s);
    if (clip > 0 && s > clip) {
        double k = clip / s;
        for (size_t i = 0; i < n; i++) g[i] *= k;
    }
    return s;
}

void opt_step(const OptCfg *o, OptState *s, const Model *m, const double *g, double lr, uint64_t *rng)
{
    double b1 = o->beta1, b2 = o->beta2, sigma = o->anneal * lr;
    s->t++;
    double c1 = 1 - pow(b1, s->t), c2 = 1 - pow(b2, s->t);
    for (size_t i = 0; i < m->n; i++) {
        double p = m->p[i], gi = g[i], wd = m->decay[i] ? o->wd : 0;
        switch (o->type) {
        case OPT_SGD: p -= lr * (gi + wd * p); break;
        case OPT_MOMENTUM: /* Nesterov */
            gi += wd * p;
            s->m[i] = b1 * s->m[i] + gi;
            p -= lr * (gi + b1 * s->m[i]);
            break;
        case OPT_ADAM:
        case OPT_ADAMW:
            if (o->type == OPT_ADAM) gi += wd * p; /* L2 inside the gradient */
            s->m[i] = b1 * s->m[i] + (1 - b1) * gi;
            s->v[i] = b2 * s->v[i] + (1 - b2) * gi * gi;
            p -= lr * (s->m[i] / c1) / (sqrt(s->v[i] / c2) + 1e-8);
            if (o->type == OPT_ADAMW) p -= lr * wd * p; /* decoupled decay */
            break;
        }
        /* annealing: random kicks that let the weights "tunnel" out of shallow minima;
         * they shrink with the learning rate, so the system cools as training goes on */
        if (sigma > 0) p += sigma * rng_normal(rng);
        m->p[i] = p;
    }
}
