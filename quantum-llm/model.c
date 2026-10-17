/* model.c - forward and backward pass of the quantum-inspired transformer.
 *
 * Gradients of complex values use the convention g = dL/dRe + i*dL/dIm, so for
 * y = a*z we get g_z = conj(a)*g_y and g_a = conj(z)*g_y. Everything is written
 * out by hand; `quantum gradcheck` verifies it against finite differences. */
#include "qcore.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPS 1e-6 /* keeps Born-rule probabilities away from exactly zero */

typedef struct {
    cpx *x, *n, *q, *k, *v, *att, *h, *cs, *m; /* cs: circuit states 0..2R+1, each T*D */
    double *xr, *hr, *a, *Z;
} Cache;

struct Work {
    int L;
    Cache *bc;
    cpx *xf, *nf, *alpha;
    double *fr, *prob;
    double *grad;
    cpx *gx, *gh, *gq, *gk, *gv, *gatt, *tmp, *galpha;
    double *gA;
};

#define CP(m, off) ((cpx *)((m)->p + (off)))
#define GP(w, off) ((cpx *)((w)->grad + (off)))

static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n ? n : 1, sz);
    if (!p) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    return p;
}

static void fill_cpx(double *p, size_t n, double std, uint64_t *rng)
{
    for (size_t i = 0; i < 2 * n; i++) p[i] = std * M_SQRT1_2 * rng_normal(rng);
}

int model_init(Model *m, Config c, uint64_t *rng)
{
    if (c.D < 2 || c.D % 2 || c.H < 1 || c.D % c.H || c.L < 1 || c.T < 2 || c.R < 0 || c.V < 2) {
        fprintf(stderr, "bad model shape: D must be even and divisible by heads\n");
        return -1;
    }
    memset(m, 0, sizeof *m);
    m->c = c;
    size_t DD = 2 * (size_t)c.D * c.D, ng = 2 * (size_t)c.R * (c.D / 2), off = 0;
    m->blk = xcalloc(c.L, sizeof *m->blk);
    m->emb = off, off += 2 * (size_t)c.V * c.D;
    for (int l = 0; l < c.L; l++) {
        BlockOff *b = &m->blk[l];
        b->wq = off, off += DD;
        b->wk = off, off += DD;
        b->wv = off, off += DD;
        b->wo = off, off += DD;
        b->theta = off, off += ng;
        b->phi = off, off += ng;
        b->psi = off, off += c.D;
        b->bias = off, off += c.D;
        b->wf = off, off += DD;
    }
    m->wout = off, off += 2 * (size_t)c.V * c.D;
    m->n = off;
    m->p = xcalloc(off, sizeof *m->p);
    m->decay = xcalloc(off, 1);

    double s = 1.0 / sqrt(c.D), so = s / sqrt(2.0 * c.L);
    fill_cpx(m->p + m->emb, (size_t)c.V * c.D, 1.0, rng);
    for (int l = 0; l < c.L; l++) {
        BlockOff *b = &m->blk[l];
        fill_cpx(m->p + b->wq, (size_t)c.D * c.D, s, rng);
        fill_cpx(m->p + b->wk, (size_t)c.D * c.D, s, rng);
        fill_cpx(m->p + b->wv, (size_t)c.D * c.D, s, rng);
        fill_cpx(m->p + b->wo, (size_t)c.D * c.D, so, rng);
        fill_cpx(m->p + b->wf, (size_t)c.D * c.D, so, rng);
        for (size_t i = 0; i < ng; i++) {
            m->p[b->theta + i] = 0.3 * rng_normal(rng);
            m->p[b->phi + i] = M_PI * (2 * rng_unif(rng) - 1);
        }
        for (int k = 0; k < c.D; k++) m->p[b->bias + k] = -0.3 + 0.1 * rng_normal(rng);
        size_t mats[] = {b->wq, b->wk, b->wv, b->wo, b->wf};
        for (int i = 0; i < 5; i++) memset(m->decay + mats[i], 1, DD);
    }
    fill_cpx(m->p + m->wout, (size_t)c.V * c.D, s, rng);
    memset(m->decay + m->wout, 1, 2 * (size_t)c.V * c.D);
    return 0;
}

void model_free(Model *m)
{
    free(m->blk);
    free(m->p);
    free(m->decay);
    memset(m, 0, sizeof *m);
}

Work *work_new(const Model *m)
{
    const Config *c = &m->c;
    size_t TD = (size_t)c->T * c->D;
    Work *w = xcalloc(1, sizeof *w);
    w->L = c->L;
    w->bc = xcalloc(c->L, sizeof *w->bc);
    for (int l = 0; l < c->L; l++) {
        Cache *b = &w->bc[l];
        b->x = xcalloc(TD, sizeof(cpx));
        b->n = xcalloc(TD, sizeof(cpx));
        b->q = xcalloc(TD, sizeof(cpx));
        b->k = xcalloc(TD, sizeof(cpx));
        b->v = xcalloc(TD, sizeof(cpx));
        b->att = xcalloc(TD, sizeof(cpx));
        b->h = xcalloc(TD, sizeof(cpx));
        b->m = xcalloc(TD, sizeof(cpx));
        b->cs = xcalloc((2 * (size_t)c->R + 2) * TD, sizeof(cpx));
        b->xr = xcalloc(c->T, sizeof(double));
        b->hr = xcalloc(c->T, sizeof(double));
        b->a = xcalloc((size_t)c->H * c->T * c->T, sizeof(double));
        b->Z = xcalloc((size_t)c->H * c->T, sizeof(double));
    }
    w->xf = xcalloc(TD, sizeof(cpx));
    w->nf = xcalloc(TD, sizeof(cpx));
    w->alpha = xcalloc((size_t)c->T * c->V, sizeof(cpx));
    w->fr = xcalloc(c->T, sizeof(double));
    w->prob = xcalloc((size_t)c->T * c->V, sizeof(double));
    w->grad = xcalloc(m->n, sizeof(double));
    w->gx = xcalloc(TD, sizeof(cpx));
    w->gh = xcalloc(TD, sizeof(cpx));
    w->gq = xcalloc(TD, sizeof(cpx));
    w->gk = xcalloc(TD, sizeof(cpx));
    w->gv = xcalloc(TD, sizeof(cpx));
    w->gatt = xcalloc(TD, sizeof(cpx));
    w->tmp = xcalloc(c->D, sizeof(cpx));
    w->galpha = xcalloc(c->V, sizeof(cpx));
    w->gA = xcalloc(c->T, sizeof(double));
    return w;
}

void work_free(Work *w)
{
    if (!w) return;
    for (int l = 0; l < w->L; l++) {
        Cache *b = &w->bc[l];
        void *ps[] = {b->x, b->n, b->q, b->k, b->v, b->att, b->h, b->m, b->cs, b->xr, b->hr, b->a, b->Z};
        for (size_t i = 0; i < sizeof ps / sizeof *ps; i++) free(ps[i]);
    }
    void *ps[] = {w->bc, w->xf, w->nf, w->alpha, w->fr, w->prob, w->grad, w->gx, w->gh,
                  w->gq, w->gk, w->gv, w->gatt, w->tmp, w->galpha, w->gA};
    for (size_t i = 0; i < sizeof ps / sizeof *ps; i++) free(ps[i]);
    free(w);
}

double *work_grad(Work *w) { return w->grad; }

/* ---------- building blocks ---------- */

/* "State preparation": scale x to a vector of length sqrt(D), like normalising a quantum state. */
static double qnorm(const cpx *x, cpx *n, int D)
{
    double r = 0;
    for (int k = 0; k < D; k++) r += creal(x[k]) * creal(x[k]) + cimag(x[k]) * cimag(x[k]);
    r = sqrt(r) + 1e-12;
    double s = sqrt((double)D) / r;
    for (int k = 0; k < D; k++) n[k] = x[k] * s;
    return r;
}

static void qnorm_back(const cpx *x, double r, const cpx *gn, cpx *gx, int D)
{
    double dot = 0;
    for (int k = 0; k < D; k++) dot += creal(conj(x[k]) * gn[k]);
    dot /= r * r;
    double s = sqrt((double)D) / r;
    for (int k = 0; k < D; k++) gx[k] += s * (gn[k] - x[k] * dot);
}

static void matvec(const cpx *W, const cpx *x, cpx *y, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        const cpx *wr = W + (size_t)i * cols;
        cpx acc = 0;
        for (int j = 0; j < cols; j++) acc += wr[j] * x[j];
        y[i] = acc;
    }
}

static void matvec_back(const cpx *W, const cpx *x, const cpx *gy, cpx *gx, cpx *gW, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        cpx g = gy[i];
        if (g == 0) continue;
        const cpx *wr = W + (size_t)i * cols;
        cpx *gr = gW + (size_t)i * cols;
        for (int j = 0; j < cols; j++) {
            gr[j] += g * conj(x[j]);
            gx[j] += conj(wr[j]) * g;
        }
    }
}

static cpx overlap(const cpx *q, const cpx *k, int n)
{
    cpx o = 0;
    for (int i = 0; i < n; i++) o += conj(q[i]) * k[i];
    return o;
}

/* Sublayer j of the brick-wall circuit: unitary 2x2 gates on pairs (a, a+1),
 * a even for even j and odd for odd j. Gate = [[c, -e^-iphi s], [e^iphi s, c]]. */
static void gates_apply(int D, int j, const double *th, const double *ph, cpx *z)
{
    int half = D / 2;
    for (int g = 0; g < half; g++) {
        int a = 2 * g + (j & 1), b = a + 1;
        if (b >= D) break;
        double c = cos(th[j * half + g]), s = sin(th[j * half + g]);
        cpx e = cexp(I * ph[j * half + g]), za = z[a], zb = z[b];
        z[a] = c * za - conj(e) * s * zb;
        z[b] = e * s * za + c * zb;
    }
}

static void gates_back(int D, int j, const double *th, const double *ph, const cpx *zin, cpx *g,
                       double *gth, double *gph)
{
    int half = D / 2;
    for (int i = 0; i < half; i++) {
        int a = 2 * i + (j & 1), b = a + 1, idx = j * half + i;
        if (b >= D) break;
        double c = cos(th[idx]), s = sin(th[idx]);
        cpx e = cexp(I * ph[idx]), za = zin[a], zb = zin[b], ga = g[a], gb = g[b];
        gth[idx] += creal(conj(ga) * (-s * za - conj(e) * c * zb) + conj(gb) * (e * c * za - s * zb));
        gph[idx] += creal(conj(ga) * (I * conj(e) * s * zb) + conj(gb) * (I * e * s * za));
        g[a] = c * ga + conj(e) * s * gb; /* the inverse (adjoint) rotation */
        g[b] = -e * s * ga + c * gb;
    }
}

static cpx pos_phase(int t, int k, int D) { return cexp(I * t * pow(10000.0, -(double)k / D)); }

/* ---------- one block: Born attention + unitary circuit ---------- */

static cpx *cstate(const Cache *c, int j, int t, int T, int D) { return c->cs + ((size_t)j * T + t) * D; }

static void block_forward(const Model *m, int l, Cache *c, cpx *xout, int n)
{
    const Config *cf = &m->c;
    const BlockOff *b = &m->blk[l];
    int D = cf->D, T = cf->T, H = cf->H, dh = D / H, J = 2 * cf->R;
    const cpx *Wq = CP(m, b->wq), *Wk = CP(m, b->wk), *Wv = CP(m, b->wv), *Wo = CP(m, b->wo), *Wf = CP(m, b->wf);
    const double *th = m->p + b->theta, *ph = m->p + b->phi, *psi = m->p + b->psi, *bias = m->p + b->bias;

    for (int t = 0; t < n; t++) {
        cpx *nt = c->n + t * D;
        c->xr[t] = qnorm(c->x + t * D, nt, D);
        matvec(Wq, nt, c->q + t * D, D, D);
        matvec(Wk, nt, c->k + t * D, D, D);
        matvec(Wv, nt, c->v + t * D, D, D);
    }
    /* attention weight of t on s (s <= t) is |<q_t|k_s>|^2, normalised like probabilities */
    for (int h = 0; h < H; h++)
        for (int t = 0; t < n; t++) {
            const cpx *q = c->q + t * D + h * dh;
            double *a = c->a + ((size_t)h * T + t) * T, Z = 0;
            for (int s = 0; s <= t; s++) {
                cpx o = overlap(q, c->k + s * D + h * dh, dh);
                a[s] = creal(o) * creal(o) + cimag(o) * cimag(o) + EPS;
                Z += a[s];
            }
            c->Z[h * T + t] = Z;
            cpx *att = c->att + t * D + h * dh;
            for (int k = 0; k < dh; k++) att[k] = 0;
            for (int s = 0; s <= t; s++) {
                a[s] /= Z;
                const cpx *v = c->v + s * D + h * dh;
                for (int k = 0; k < dh; k++) att[k] += a[s] * v[k];
            }
        }
    for (int t = 0; t < n; t++) {
        cpx *ht = c->h + t * D, *z = cstate(c, 0, t, T, D);
        matvec(Wo, c->att + t * D, ht, D, D);
        for (int k = 0; k < D; k++) ht[k] += c->x[t * D + k];
        c->hr[t] = qnorm(ht, z, D);
        for (int j = 0; j < J; j++) {
            cpx *zn = cstate(c, j + 1, t, T, D);
            memcpy(zn, z, D * sizeof *z);
            gates_apply(D, j, th, ph, zn);
            z = zn;
        }
        cpx *u = cstate(c, J + 1, t, T, D), *mt = c->m + t * D;
        for (int k = 0; k < D; k++) {
            u[k] = cexp(I * psi[k]) * z[k];
            double r = cabs(u[k]);
            mt[k] = (r > 1e-12 && r + bias[k] > 0) ? u[k] * ((r + bias[k]) / r) : 0; /* modReLU */
        }
        matvec(Wf, mt, xout + t * D, D, D);
        for (int k = 0; k < D; k++) xout[t * D + k] += ht[k];
    }
}

/* On entry w->gx holds dL/d(block output); on exit dL/d(block input). */
static void block_backward(const Model *m, int l, const Cache *c, Work *w, int n)
{
    const Config *cf = &m->c;
    const BlockOff *b = &m->blk[l];
    int D = cf->D, T = cf->T, H = cf->H, dh = D / H, J = 2 * cf->R;
    const cpx *Wq = CP(m, b->wq), *Wk = CP(m, b->wk), *Wv = CP(m, b->wv), *Wo = CP(m, b->wo), *Wf = CP(m, b->wf);
    const double *th = m->p + b->theta, *ph = m->p + b->phi, *psi = m->p + b->psi, *bias = m->p + b->bias;
    cpx *gWq = GP(w, b->wq), *gWk = GP(w, b->wk), *gWv = GP(w, b->wv), *gWo = GP(w, b->wo), *gWf = GP(w, b->wf);
    double *gth = w->grad + b->theta, *gph = w->grad + b->phi, *gpsi = w->grad + b->psi, *gbias = w->grad + b->bias;
    size_t bytes = (size_t)n * D * sizeof(cpx);
    cpx *gx = w->gx, *gh = w->gh, *g = w->tmp;

    /* feed-forward branch: x_out = h + Wf * modReLU(phases * gates * qnorm(h)) */
    memcpy(gh, gx, bytes);
    for (int t = 0; t < n; t++) {
        memset(g, 0, D * sizeof *g);
        matvec_back(Wf, c->m + t * D, gx + t * D, g, gWf, D, D);
        const cpx *u = cstate(c, J + 1, t, T, D);
        for (int k = 0; k < D; k++) {
            double r = cabs(u[k]);
            if (r > 1e-12 && r + bias[k] > 0) {
                cpx e = u[k] / r;
                double dot = creal(conj(e) * g[k]);
                gbias[k] += dot;
                g[k] += bias[k] * (g[k] - e * dot) / r;
            } else
                g[k] = 0;
            gpsi[k] += creal(conj(g[k]) * I * u[k]);
            g[k] *= cexp(-I * psi[k]);
        }
        for (int j = J - 1; j >= 0; j--) gates_back(D, j, th, ph, cstate(c, j, t, T, D), g, gth, gph);
        qnorm_back(c->h + t * D, c->hr[t], g, gh + t * D, D);
    }

    /* attention branch: h = x + Wo * att */
    memcpy(gx, gh, bytes);
    memset(w->gq, 0, bytes);
    memset(w->gk, 0, bytes);
    memset(w->gv, 0, bytes);
    memset(w->gatt, 0, bytes);
    for (int t = 0; t < n; t++) matvec_back(Wo, c->att + t * D, gh + t * D, w->gatt + t * D, gWo, D, D);
    for (int h = 0; h < H; h++)
        for (int t = 0; t < n; t++) {
            const cpx *q = c->q + t * D + h * dh, *gat = w->gatt + t * D + h * dh;
            const double *a = c->a + ((size_t)h * T + t) * T;
            double Z = c->Z[h * T + t], dot = 0;
            for (int s = 0; s <= t; s++) {
                const cpx *v = c->v + s * D + h * dh;
                cpx *gv = w->gv + s * D + h * dh;
                double ga = 0;
                for (int k = 0; k < dh; k++) {
                    ga += creal(conj(gat[k]) * v[k]);
                    gv[k] += a[s] * gat[k];
                }
                w->gA[s] = ga;
                dot += a[s] * ga;
            }
            for (int s = 0; s <= t; s++) {
                const cpx *kk = c->k + s * D + h * dh;
                cpx go = 2.0 * overlap(q, kk, dh) * ((w->gA[s] - dot) / Z);
                cpx *gk = w->gk + s * D + h * dh, *gq = w->gq + t * D + h * dh;
                for (int k = 0; k < dh; k++) {
                    gk[k] += q[k] * go;
                    gq[k] += kk[k] * conj(go);
                }
            }
        }
    for (int t = 0; t < n; t++) {
        const cpx *nt = c->n + t * D;
        memset(g, 0, D * sizeof *g);
        matvec_back(Wq, nt, w->gq + t * D, g, gWq, D, D);
        matvec_back(Wk, nt, w->gk + t * D, g, gWk, D, D);
        matvec_back(Wv, nt, w->gv + t * D, g, gWv, D, D);
        qnorm_back(c->x + t * D, c->xr[t], g, gx + t * D, D);
    }
}

/* ---------- whole model ---------- */

double model_forward(const Model *m, Work *w, const int *tok, const int *tgt, int n)
{
    const Config *c = &m->c;
    int D = c->D, V = c->V;
    const cpx *E = CP(m, m->emb), *Wout = CP(m, m->wout);
    cpx *x = w->bc[0].x;
    for (int t = 0; t < n; t++)
        for (int k = 0; k < D; k++) x[t * D + k] = E[(size_t)tok[t] * D + k] * pos_phase(t, k, D);
    for (int l = 0; l < c->L; l++) block_forward(m, l, &w->bc[l], l + 1 < c->L ? w->bc[l + 1].x : w->xf, n);

    double loss = 0;
    for (int t = 0; t < n; t++) {
        cpx *al = w->alpha + (size_t)t * V;
        double *p = w->prob + (size_t)t * V, S = 0;
        w->fr[t] = qnorm(w->xf + t * D, w->nf + t * D, D);
        matvec(Wout, w->nf + t * D, al, V, D);
        for (int v = 0; v < V; v++) S += p[v] = creal(al[v]) * creal(al[v]) + cimag(al[v]) * cimag(al[v]) + EPS;
        for (int v = 0; v < V; v++) p[v] /= S; /* Born rule */
        if (tgt) loss -= log(p[tgt[t]]);
    }
    return tgt ? loss / n : 0;
}

void model_backward(const Model *m, Work *w, const int *tok, const int *tgt, int n, double scale)
{
    const Config *c = &m->c;
    int D = c->D, V = c->V;
    const cpx *Wout = CP(m, m->wout);
    cpx *gE = GP(w, m->emb), *gWout = GP(w, m->wout);
    double sc = scale / n;

    memset(w->gx, 0, (size_t)n * D * sizeof(cpx));
    for (int t = 0; t < n; t++) {
        const cpx *al = w->alpha + (size_t)t * V;
        double S = 0;
        for (int v = 0; v < V; v++) S += creal(al[v]) * creal(al[v]) + cimag(al[v]) * cimag(al[v]) + EPS;
        int y = tgt[t];
        double sy = creal(al[y]) * creal(al[y]) + cimag(al[y]) * cimag(al[y]) + EPS;
        for (int v = 0; v < V; v++) w->galpha[v] = 2.0 * al[v] * (sc * (1.0 / S - (v == y ? 1.0 / sy : 0.0)));
        memset(w->tmp, 0, D * sizeof(cpx));
        matvec_back(Wout, w->nf + t * D, w->galpha, w->tmp, gWout, V, D);
        qnorm_back(w->xf + t * D, w->fr[t], w->tmp, w->gx + t * D, D);
    }
    for (int l = c->L - 1; l >= 0; l--) block_backward(m, l, &w->bc[l], w, n);
    for (int t = 0; t < n; t++)
        for (int k = 0; k < D; k++) gE[(size_t)tok[t] * D + k] += conj(pos_phase(t, k, D)) * w->gx[t * D + k];
}

void model_inspect(const Model *m, Work *w, const int *tok, int n, const unsigned char *itos, Inspect *ins)
{
    const Config *c = &m->c;
    int D = c->D, T = c->T, J = 2 * c->R;
    if (n > INS_T) tok += n - INS_T, n = INS_T;
    if (n > T) tok += n - T, n = T;
    memset(ins, 0, sizeof *ins);
    if (n < 1) return;
    model_forward(m, w, tok, NULL, n);
    ins->n = n;
    ins->L = c->L < INS_L ? c->L : INS_L;
    for (int t = 0; t < n; t++) {
        ins->ch[t] = itos[tok[t]];
        for (int k = 0; k < 4 && k < D; k++) {
            cpx x = w->bc[0].x[t * D + k];
            ins->emb_amp[t][k] = cabs(x);
            ins->emb_ph[t][k] = carg(x);
        }
    }
    for (int l = 0; l < ins->L; l++) {
        const Cache *b = &w->bc[l];
        for (int h = 0; h < c->H; h++)
            for (int t = 0; t < n; t++)
                for (int s = 0; s <= t; s++) ins->att[l][t][s] += b->a[((size_t)h * T + t) * T + s] / c->H;
        for (int k = 0; k < INS_N && k < D; k++) {
            cpx v = b->m[(n - 1) * D + k];
            ins->act_amp[l][k] = cabs(v);
            ins->act_ph[l][k] = carg(v);
            ins->gate[l][k] = J > 0 && k < D / 2 ? m->p[m->blk[l].theta + k] : 0;
        }
    }
    const double *p = w->prob + (size_t)(n - 1) * c->V;
    for (int i = 0; i < 8; i++) {
        int best = -1;
        for (int v = 0; v < c->V; v++) {
            int used = 0;
            for (int j = 0; j < i; j++) used |= ins->top_ch[j] == itos[v];
            if (!used && (best < 0 || p[v] > p[best])) best = v;
        }
        ins->top_ch[i] = best < 0 ? ' ' : itos[best];
        ins->top_p[i] = best < 0 ? 0 : p[best];
    }
}

int model_sample(const Model *m, Work *w, const int *ctx, int n, double temp, uint64_t *rng)
{
    int V = m->c.V;
    model_forward(m, w, ctx, NULL, n);
    const double *p = w->prob + (size_t)(n - 1) * V;
    if (temp <= 0) { /* greedy */
        int best = 0;
        for (int v = 1; v < V; v++)
            if (p[v] > p[best]) best = v;
        return best;
    }
    double S = 0, wt[256]; /* V <= 256: the vocabulary is bytes */
    for (int v = 0; v < V; v++) S += wt[v] = pow(p[v], 1.0 / temp);
    double r = rng_unif(rng) * S;
    for (int v = 0; v < V; v++)
        if ((r -= wt[v]) <= 0) return v;
    return V - 1;
}
