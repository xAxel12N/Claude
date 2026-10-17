/* gpu.c - the QuantumGPT forward pass, backward pass and optimiser as OpenCL kernels.
 * Mirrors model.c and optim.c step by step, in 32-bit floats. libOpenCL is loaded at run
 * time, so building needs no OpenCL headers or libraries; the GPU driver provides it. */
#include "gpu.h"
#include <dlfcn.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- the bits of the OpenCL API we use ---------- */

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_bitfield;
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;

#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_DEVICE_NAME 0x102B
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_MEM_READ_WRITE (1 << 0)

static struct {
    void *lib;
    cl_int (*GetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
    cl_int (*GetDeviceIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id *, cl_uint *);
    cl_int (*GetDeviceInfo)(cl_device_id, cl_uint, size_t, void *, size_t *);
    cl_context (*CreateContext)(const intptr_t *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
    cl_command_queue (*CreateCommandQueue)(cl_context, cl_device_id, cl_bitfield, cl_int *);
    cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
    cl_int (*BuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
    cl_int (*GetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
    cl_kernel (*CreateKernel)(cl_program, const char *, cl_int *);
    cl_mem (*CreateBuffer)(cl_context, cl_bitfield, size_t, void *, cl_int *);
    cl_int (*SetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
    cl_int (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *,
                                   const size_t *, cl_uint, const void *, void *);
    cl_int (*EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void *, cl_uint,
                                 const void *, void *);
    cl_int (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void *, cl_uint, const void *,
                                void *);
    cl_int (*Finish)(cl_command_queue);
    cl_int (*ReleaseMemObject)(cl_mem);
    cl_int (*ReleaseKernel)(cl_kernel);
    cl_int (*ReleaseProgram)(cl_program);
    cl_int (*ReleaseCommandQueue)(cl_command_queue);
    cl_int (*ReleaseContext)(cl_context);
} cl;

static int load_opencl(char *err, size_t errn)
{
    if (cl.lib) return 0;
    const char *names[] = {"libOpenCL.so.1", "libOpenCL.so"};
    for (int i = 0; i < 2 && !cl.lib; i++) cl.lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
    if (!cl.lib) {
        snprintf(err, errn, "OpenCL is not installed (sudo apt install ocl-icd-libopencl1)");
        return -1;
    }
#define LOAD(f)                                                                                                        \
    if (!(*(void **)&cl.f = dlsym(cl.lib, "cl" #f))) {                                                                 \
        snprintf(err, errn, "OpenCL library lacks cl" #f);                                                             \
        return -1;                                                                                                     \
    }
    LOAD(GetPlatformIDs) LOAD(GetDeviceIDs) LOAD(GetDeviceInfo) LOAD(CreateContext) LOAD(CreateCommandQueue)
    LOAD(CreateProgramWithSource) LOAD(BuildProgram) LOAD(GetProgramBuildInfo) LOAD(CreateKernel) LOAD(CreateBuffer)
    LOAD(SetKernelArg) LOAD(EnqueueNDRangeKernel) LOAD(EnqueueWriteBuffer) LOAD(EnqueueReadBuffer) LOAD(Finish)
    LOAD(ReleaseMemObject) LOAD(ReleaseKernel) LOAD(ReleaseProgram) LOAD(ReleaseCommandQueue) LOAD(ReleaseContext)
#undef LOAD
    return 0;
}

/* ---------- kernels ---------- */

#define CLSRC(...) #__VA_ARGS__
static const char *SRC = CLSRC(
typedef float2 cf;
cf cmul(cf a, cf b) { return (cf)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
cf cconj(cf a) { return (cf)(a.x, -a.y); }
cf cis(float t) { return (cf)(cos(t), sin(t)); }
float cabs2(cf a) { return a.x * a.x + a.y * a.y; }
float cdot(cf a, cf b) { return a.x * b.x + a.y * b.y; }
cf ci(cf a) { return (cf)(-a.y, a.x); }

kernel void fill(global float *b, int n) {
    int i = get_global_id(0);
    if (i < n) b[i] = 0;
}

kernel void copy(global const cf *a, global cf *b, int n) {
    int i = get_global_id(0);
    if (i < n) b[i] = a[i];
}

kernel void embed(global const int *tok, global const float *P, int emb, global cf *X, int T, int D, int n) {
    int g = get_global_id(0);
    if (g >= n) return;
    int row = g / D, k = g % D;
    global const cf *E = (global const cf *)(P + emb);
    X[g] = cmul(E[tok[row] * D + k], cis((row % T) * pow(10000.0f, -(float)k / D)));
}

kernel void embed_back(global const int *tok, global const cf *gX, global float *G, int emb, int N, int T, int D,
                       int V) {
    int g = get_global_id(0);
    if (g >= V * D) return;
    int v = g / D, k = g % D;
    float om = pow(10000.0f, -(float)k / D);
    cf acc = (cf)(0, 0);
    for (int r = 0; r < N; r++)
        if (tok[r] == v) acc += cmul(cconj(cis((r % T) * om)), gX[r * D + k]);
    global cf *gE = (global cf *)(G + emb);
    gE[g] += acc;
}

kernel void qnorm(global const cf *X, global cf *Y, global float *r, int D, int N) {
    int row = get_global_id(0);
    if (row >= N) return;
    global const cf *x = X + row * D;
    float s = 0;
    for (int k = 0; k < D; k++) s += cabs2(x[k]);
    float rr = sqrt(s) + 1e-12f, sc = sqrt((float)D) / rr;
    r[row] = rr;
    for (int k = 0; k < D; k++) Y[row * D + k] = x[k] * sc;
}

kernel void qnorm_back(global const cf *X, global const float *r, global const cf *gY, global cf *gX, int D, int N,
                       int accum) {
    int row = get_global_id(0);
    if (row >= N) return;
    global const cf *x = X + row * D;
    global const cf *g = gY + row * D;
    float rr = r[row], dot = 0;
    for (int k = 0; k < D; k++) dot += cdot(x[k], g[k]);
    dot /= rr * rr;
    float sc = sqrt((float)D) / rr;
    for (int k = 0; k < D; k++) {
        cf v = sc * (g[k] - x[k] * dot);
        gX[row * D + k] = accum ? gX[row * D + k] + v : v;
    }
}

kernel void mm(global const float *P, int w, global const cf *X, global cf *Y, global const cf *Add, int R, int C,
               int N, int add) {
    int g = get_global_id(0);
    if (g >= N * R) return;
    int n = g / R, i = g % R;
    global const cf *W = (global const cf *)(P + w) + i * C;
    global const cf *x = X + n * C;
    cf acc = add ? Add[g] : (cf)(0, 0);
    for (int j = 0; j < C; j++) acc += cmul(W[j], x[j]);
    Y[g] = acc;
}

kernel void mm_bx(global const float *P, int w, global const cf *gY, global cf *gX, int R, int C, int N, int accum) {
    int g = get_global_id(0);
    if (g >= N * C) return;
    int n = g / C, j = g % C;
    global const cf *W = (global const cf *)(P + w);
    cf acc = (cf)(0, 0);
    for (int i = 0; i < R; i++) acc += cmul(cconj(W[i * C + j]), gY[n * R + i]);
    gX[g] = accum ? gX[g] + acc : acc;
}

kernel void mm_bw(global const cf *gY, global const cf *X, global float *G, int w, int R, int C, int N) {
    int g = get_global_id(0);
    if (g >= R * C) return;
    int i = g / C, j = g % C;
    cf acc = (cf)(0, 0);
    for (int n = 0; n < N; n++) acc += cmul(gY[n * R + i], cconj(X[n * C + j]));
    global cf *gW = (global cf *)(G + w);
    gW[g] += acc;
}

kernel void attn(global const cf *q, global const cf *k, global const cf *v, global float *A, global float *Z,
                 global cf *att, int B, int T, int H, int D) {
    int g = get_global_id(0);
    if (g >= B * H * T) return;
    int t = g % T, h = (g / T) % H, b = g / (T * H), dh = D / H;
    global const cf *qt = q + (b * T + t) * D + h * dh;
    global float *a = A + (size_t)g * T;
    float z = 0;
    for (int s = 0; s <= t; s++) {
        global const cf *ks = k + (b * T + s) * D + h * dh;
        cf o = (cf)(0, 0);
        for (int i = 0; i < dh; i++) o += cmul(cconj(qt[i]), ks[i]);
        a[s] = cabs2(o) + 1e-6f;
        z += a[s];
    }
    Z[g] = z;
    for (int s = 0; s <= t; s++) a[s] /= z;
    global cf *at = att + (b * T + t) * D + h * dh;
    for (int i = 0; i < dh; i++) {
        cf acc = (cf)(0, 0);
        for (int s = 0; s <= t; s++) acc += a[s] * v[(b * T + s) * D + h * dh + i];
        at[i] = acc;
    }
}

kernel void attn_bq(global const cf *q, global const cf *k, global const cf *v, global const float *A,
                    global const float *Z, global const cf *gatt, global cf *gO, global cf *gq, int B, int T, int H,
                    int D) {
    int g = get_global_id(0);
    if (g >= B * H * T) return;
    int t = g % T, h = (g / T) % H, b = g / (T * H), dh = D / H;
    global const cf *qt = q + (b * T + t) * D + h * dh;
    global const cf *gat = gatt + (b * T + t) * D + h * dh;
    global const float *a = A + (size_t)g * T;
    global cf *out = gq + (b * T + t) * D + h * dh;
    float z = Z[g], dot = 0;
    for (int s = 0; s <= t; s++) {
        global const cf *vs = v + (b * T + s) * D + h * dh;
        float ga = 0;
        for (int i = 0; i < dh; i++) ga += cdot(gat[i], vs[i]);
        dot += a[s] * ga;
    }
    for (int i = 0; i < dh; i++) out[i] = (cf)(0, 0);
    for (int s = 0; s <= t; s++) {
        global const cf *vs = v + (b * T + s) * D + h * dh;
        global const cf *ks = k + (b * T + s) * D + h * dh;
        float ga = 0;
        cf o = (cf)(0, 0);
        for (int i = 0; i < dh; i++) ga += cdot(gat[i], vs[i]), o += cmul(cconj(qt[i]), ks[i]);
        cf go = o * (2.0f * (ga - dot) / z);
        gO[(size_t)g * T + s] = go;
        for (int i = 0; i < dh; i++) out[i] += cmul(ks[i], cconj(go));
    }
}

kernel void attn_bkv(global const cf *q, global const float *A, global const cf *gatt, global const cf *gO,
                     global cf *gk, global cf *gv, int B, int T, int H, int D) {
    int g = get_global_id(0);
    if (g >= B * H * T) return;
    int s = g % T, h = (g / T) % H, b = g / (T * H), dh = D / H;
    global cf *ok = gk + (b * T + s) * D + h * dh;
    global cf *ov = gv + (b * T + s) * D + h * dh;
    for (int i = 0; i < dh; i++) ok[i] = (cf)(0, 0), ov[i] = (cf)(0, 0);
    for (int t = s; t < T; t++) {
        size_t row = ((size_t)(b * H + h) * T + t) * T + s;
        cf go = gO[row];
        float a = A[row];
        global const cf *qt = q + (b * T + t) * D + h * dh;
        global const cf *gat = gatt + (b * T + t) * D + h * dh;
        for (int i = 0; i < dh; i++) ok[i] += cmul(qt[i], go), ov[i] += a * gat[i];
    }
}

kernel void ff(global const float *P, global const cf *Hh, global float *hr, global cf *CS, global cf *M, int th,
               int ph, int psi, int bias, int D, int J, int N) {
    int row = get_global_id(0);
    if (row >= N) return;
    global const cf *h = Hh + row * D;
    global cf *z = CS + (size_t)row * D;
    float s = 0;
    for (int k = 0; k < D; k++) s += cabs2(h[k]);
    float rr = sqrt(s) + 1e-12f, sc = sqrt((float)D) / rr;
    hr[row] = rr;
    for (int k = 0; k < D; k++) z[k] = h[k] * sc;
    int hf = D / 2;
    for (int j = 0; j < J; j++) {
        global cf *zn = CS + ((size_t)(j + 1) * N + row) * D;
        for (int k = 0; k < D; k++) zn[k] = z[k];
        for (int gi = 0; gi < hf; gi++) {
            int a = 2 * gi + (j & 1), b = a + 1;
            if (b >= D) break;
            float c = cos(P[th + j * hf + gi]), sn = sin(P[th + j * hf + gi]);
            cf e = cis(P[ph + j * hf + gi]), za = zn[a], zb = zn[b];
            zn[a] = c * za - sn * cmul(cconj(e), zb);
            zn[b] = sn * cmul(e, za) + c * zb;
        }
        z = zn;
    }
    global cf *u = CS + ((size_t)(J + 1) * N + row) * D;
    for (int k = 0; k < D; k++) {
        u[k] = cmul(cis(P[psi + k]), z[k]);
        float r = length(u[k]), bk = P[bias + k];
        M[row * D + k] = (r > 1e-12f && r + bk > 0) ? u[k] * ((r + bk) / r) : (cf)(0, 0);
    }
}

kernel void ff_back(global const float *P, global const cf *Hh, global const float *hr, global const cf *CS,
                    global cf *gM, global cf *gH, global float *pg, int th, int ph, int psi, int bias, int D, int J,
                    int N) {
    int row = get_global_id(0);
    if (row >= N) return;
    int hf = D / 2, M = 2 * J * hf + 2 * D;
    global cf *g = gM + row * D;
    global float *pth = pg + (size_t)row * M, *pph = pth + J * hf, *ppsi = pph + J * hf, *pb = ppsi + D;
    global const cf *u = CS + ((size_t)(J + 1) * N + row) * D;
    for (int k = 0; k < D; k++) {
        float r = length(u[k]), bk = P[bias + k];
        cf gk = g[k];
        if (r > 1e-12f && r + bk > 0) {
            cf e = u[k] / r;
            float dot = cdot(e, gk);
            pb[k] = dot;
            gk = gk + bk * (gk - e * dot) / r;
        } else {
            pb[k] = 0;
            gk = (cf)(0, 0);
        }
        ppsi[k] = cdot(gk, ci(u[k]));
        g[k] = cmul(gk, cis(-P[psi + k]));
    }
    for (int i = 0; i < J * hf; i++) pth[i] = 0, pph[i] = 0;
    for (int j = J - 1; j >= 0; j--) {
        global const cf *zin = CS + ((size_t)j * N + row) * D;
        for (int gi = 0; gi < hf; gi++) {
            int a = 2 * gi + (j & 1), b = a + 1, idx = j * hf + gi;
            if (b >= D) break;
            float c = cos(P[th + idx]), sn = sin(P[th + idx]);
            cf e = cis(P[ph + idx]), ec = cconj(e), za = zin[a], zb = zin[b], ga = g[a], gb = g[b];
            pth[idx] = cdot(ga, -sn * za - c * cmul(ec, zb)) + cdot(gb, c * cmul(e, za) - sn * zb);
            pph[idx] = cdot(ga, sn * ci(cmul(ec, zb))) + cdot(gb, sn * ci(cmul(e, za)));
            g[a] = c * ga + sn * cmul(ec, gb);
            g[b] = -sn * cmul(e, ga) + c * gb;
        }
    }
    global const cf *h = Hh + row * D;
    float rr = hr[row], dot = 0;
    for (int k = 0; k < D; k++) dot += cdot(h[k], g[k]);
    dot /= rr * rr;
    float sc = sqrt((float)D) / rr;
    for (int k = 0; k < D; k++) gH[row * D + k] += sc * (g[k] - h[k] * dot);
}

kernel void reduce_rows(global const float *pg, int M, int col0, int cnt, global float *G, int off, int N) {
    int i = get_global_id(0);
    if (i >= cnt) return;
    float s = 0;
    for (int n = 0; n < N; n++) s += pg[(size_t)n * M + col0 + i];
    G[off + i] += s;
}

kernel void born(global const cf *al, global const int *tgt, global float *loss, global cf *gal, int V, int N,
                 float scale, int grad) {
    int row = get_global_id(0);
    if (row >= N) return;
    global const cf *a = al + (size_t)row * V;
    float S = 0;
    for (int v = 0; v < V; v++) S += cabs2(a[v]) + 1e-6f;
    int y = tgt[row];
    float sy = cabs2(a[y]) + 1e-6f;
    loss[row] = log(S) - log(sy);
    if (grad)
        for (int v = 0; v < V; v++) gal[(size_t)row * V + v] = a[v] * (2.0f * scale * (1.0f / S - (v == y ? 1.0f / sy : 0.0f)));
}

kernel void sumsq(global const float *G, int n, global float *part, int chunks) {
    int c = get_global_id(0);
    if (c >= chunks) return;
    float s = 0;
    for (int i = c; i < n; i += chunks) s += G[i] * G[i];
    part[c] = s;
}

uint mix(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

kernel void opt(global float *P, global const float *G, global float *Mm, global float *Vv, global const uchar *decay,
                int n, int type, float lr, float b1, float b2, float c1, float c2, float wd, float gscale,
                float sigma, uint seed) {
    int i = get_global_id(0);
    if (i >= n) return;
    float p = P[i], gi = G[i] * gscale, w = decay[i] ? wd : 0;
    if (type == 0) p -= lr * (gi + w * p);
    else if (type == 1) {
        gi += w * p;
        Mm[i] = b1 * Mm[i] + gi;
        p -= lr * (gi + b1 * Mm[i]);
    } else {
        if (type == 2) gi += w * p;
        Mm[i] = b1 * Mm[i] + (1 - b1) * gi;
        Vv[i] = b2 * Vv[i] + (1 - b2) * gi * gi;
        p -= lr * (Mm[i] / c1) / (sqrt(Vv[i] / c2) + 1e-8f);
        if (type == 3) p -= lr * w * p;
    }
    if (sigma > 0) {
        float u1 = (mix(2 * i + seed) + 1.0f) / 4294967296.0f, u2 = mix(2 * i + 1 + seed * 7919u) / 4294967296.0f;
        p += sigma * sqrt(-2.0f * log(u1)) * cos(6.2831853f * u2);
    }
    P[i] = p;
}
);

/* ---------- host side ---------- */

enum { K_FILL, K_COPY, K_EMBED, K_EMBED_BACK, K_QNORM, K_QNORM_BACK, K_MM, K_MM_BX, K_MM_BW, K_ATTN, K_ATTN_BQ,
       K_ATTN_BKV, K_FF, K_FF_BACK, K_REDUCE, K_BORN, K_SUMSQ, K_OPT, K_COUNT };
static const char *KNAMES[K_COUNT] = {"fill", "copy", "embed", "embed_back", "qnorm", "qnorm_back", "mm", "mm_bx",
                                      "mm_bw", "attn", "attn_bq", "attn_bkv", "ff", "ff_back", "reduce_rows",
                                      "born", "sumsq", "opt"};

#define MAXL 16
#define CHUNKS 1024

typedef struct {
    cl_mem x, n, xr, q, k, v, A, Z, att, h, hr, cs, m;
} Layer;

struct Gpu {
    cl_device_id dev;
    cl_context ctx;
    cl_command_queue q;
    cl_program prog;
    cl_kernel k[K_COUNT];
    char name[128], err[512];
    /* bound model */
    Config c;
    int B, L;
    size_t n;
    cl_mem P, G, Mm, Vv, decay, tok, tgt, loss, xf, nf, fr, alpha, galpha, gx, gh, gq, gk, gv, gatt, gtmp, gO, pg,
        part;
    Layer ly[MAXL];
    float *host; /* staging for conversions, n floats */
};

const char *gpu_name(const Gpu *g) { return g->name; }
const char *gpu_error(const Gpu *g) { return g->err; }

static int fail(Gpu *g, const char *what, cl_int e)
{
    snprintf(g->err, sizeof g->err, "%s failed (OpenCL error %d)", what, e);
    return -1;
}

static void release_buffers(Gpu *g)
{
    cl_mem *all[] = {&g->P, &g->G, &g->Mm, &g->Vv, &g->decay, &g->tok, &g->tgt, &g->loss, &g->xf, &g->nf,
                     &g->fr, &g->alpha, &g->galpha, &g->gx, &g->gh, &g->gq, &g->gk, &g->gv, &g->gatt, &g->gtmp,
                     &g->gO, &g->pg, &g->part};
    for (size_t i = 0; i < sizeof all / sizeof *all; i++)
        if (*all[i]) cl.ReleaseMemObject(*all[i]), *all[i] = NULL;
    for (int l = 0; l < MAXL; l++) {
        cl_mem *b = (cl_mem *)&g->ly[l];
        for (size_t i = 0; i < sizeof(Layer) / sizeof(cl_mem); i++)
            if (b[i]) cl.ReleaseMemObject(b[i]), b[i] = NULL;
    }
    free(g->host);
    g->host = NULL;
    g->B = 0;
}

void gpu_close(Gpu *g)
{
    if (!g) return;
    release_buffers(g);
    for (int i = 0; i < K_COUNT; i++)
        if (g->k[i]) cl.ReleaseKernel(g->k[i]);
    if (g->prog) cl.ReleaseProgram(g->prog);
    if (g->q) cl.ReleaseCommandQueue(g->q);
    if (g->ctx) cl.ReleaseContext(g->ctx);
    free(g);
}

Gpu *gpu_open(char *err, size_t errn)
{
    if (load_opencl(err, errn)) return NULL;
    cl_platform_id plats[16];
    cl_uint np = 0;
    if (cl.GetPlatformIDs(16, plats, &np) || np == 0) {
        snprintf(err, errn, "no OpenCL driver found (for NVIDIA: the driver's OpenCL part, e.g. libnvidia-compute)");
        return NULL;
    }
    int any = getenv("QGPT_OPENCL") && !strcmp(getenv("QGPT_OPENCL"), "any");
    cl_device_id dev = NULL;
    for (int pass = 0; pass < 2 && !dev; pass++) {
        if (pass == 1 && !any) break;
        for (cl_uint i = 0; i < np && !dev; i++) {
            cl_uint nd = 0;
            if (cl.GetDeviceIDs(plats[i], pass ? CL_DEVICE_TYPE_ALL : CL_DEVICE_TYPE_GPU, 1, &dev, &nd) || !nd)
                dev = NULL;
        }
    }
    if (!dev) {
        snprintf(err, errn, "no OpenCL graphics card found");
        return NULL;
    }
    Gpu *g = calloc(1, sizeof *g);
    g->dev = dev;
    cl.GetDeviceInfo(dev, CL_DEVICE_NAME, sizeof g->name - 1, g->name, NULL);
    cl_int e;
    g->ctx = cl.CreateContext(NULL, 1, &dev, NULL, NULL, &e);
    if (!g->ctx) goto bad;
    g->q = cl.CreateCommandQueue(g->ctx, dev, 0, &e);
    if (!g->q) goto bad;
    g->prog = cl.CreateProgramWithSource(g->ctx, 1, &SRC, NULL, &e);
    if (!g->prog) goto bad;
    if ((e = cl.BuildProgram(g->prog, 1, &dev, "", NULL, NULL))) {
        char log[4096] = "";
        cl.GetProgramBuildInfo(g->prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log - 1, log, NULL);
        snprintf(err, errn, "GPU program did not compile: %.400s", log);
        gpu_close(g);
        return NULL;
    }
    for (int i = 0; i < K_COUNT; i++)
        if (!(g->k[i] = cl.CreateKernel(g->prog, KNAMES[i], &e))) goto bad;
    return g;
bad:
    snprintf(err, errn, "OpenCL setup failed (error %d)", e);
    gpu_close(g);
    return NULL;
}

/* run(g, kernel, global size, "mif...", args): m = cl_mem, i = int, f = float, u = uint */
static int run(Gpu *g, int k, size_t n, const char *sig, ...)
{
    va_list ap;
    va_start(ap, sig);
    cl_int e = 0;
    for (cl_uint i = 0; sig[i] && !e; i++) {
        if (sig[i] == 'm') {
            cl_mem m = va_arg(ap, cl_mem);
            e = cl.SetKernelArg(g->k[k], i, sizeof m, &m);
        } else if (sig[i] == 'i') {
            int v = va_arg(ap, int);
            e = cl.SetKernelArg(g->k[k], i, sizeof v, &v);
        } else if (sig[i] == 'f') {
            float v = (float)va_arg(ap, double);
            e = cl.SetKernelArg(g->k[k], i, sizeof v, &v);
        } else {
            cl_uint v = va_arg(ap, unsigned);
            e = cl.SetKernelArg(g->k[k], i, sizeof v, &v);
        }
    }
    va_end(ap);
    if (e) return fail(g, KNAMES[k], e);
    if (n == 0) return 0;
    if ((e = cl.EnqueueNDRangeKernel(g->q, g->k[k], 1, NULL, &n, NULL, 0, NULL, NULL))) return fail(g, KNAMES[k], e);
    return 0;
}

static cl_mem buf(Gpu *g, size_t bytes, int *bad)
{
    cl_int e;
    cl_mem m = cl.CreateBuffer(g->ctx, CL_MEM_READ_WRITE, bytes ? bytes : 4, NULL, &e);
    if (!m) *bad = 1;
    return m;
}

static int put(Gpu *g, cl_mem m, const void *src, size_t bytes)
{
    cl_int e = cl.EnqueueWriteBuffer(g->q, m, 1, 0, bytes, src, 0, NULL, NULL);
    return e ? fail(g, "upload", e) : 0;
}

static int get(Gpu *g, cl_mem m, void *dst, size_t bytes)
{
    cl_int e = cl.EnqueueReadBuffer(g->q, m, 1, 0, bytes, dst, 0, NULL, NULL);
    return e ? fail(g, "download", e) : 0;
}

static int upload_doubles(Gpu *g, cl_mem m, const double *src, size_t n)
{
    for (size_t i = 0; i < n; i++) g->host[i] = (float)src[i];
    return put(g, m, g->host, n * sizeof(float));
}

int gpu_bind(Gpu *g, const Model *m, const OptState *s, int B)
{
    const Config *c = &m->c;
    if (c->L > MAXL || c->D > 256) {
        snprintf(g->err, sizeof g->err, "brain too big for the GPU path (max %d blocks, width 256)", MAXL);
        return -1;
    }
    release_buffers(g);
    g->c = *c, g->B = B, g->L = c->L, g->n = m->n;
    size_t N = (size_t)B * c->T, D = c->D, cfb = 8, ND = N * D * cfb, J = 2 * c->R;
    size_t att = (size_t)B * c->H * c->T * c->T, pgM = 2 * J * (D / 2) + 2 * D;
    int bad = 0;
    size_t hn = m->n > N ? m->n : N;
    g->host = malloc((hn > CHUNKS ? hn : CHUNKS) * sizeof(float));
    g->P = buf(g, m->n * 4, &bad), g->G = buf(g, m->n * 4, &bad);
    g->Mm = buf(g, m->n * 4, &bad), g->Vv = buf(g, m->n * 4, &bad), g->decay = buf(g, m->n, &bad);
    g->tok = buf(g, N * 4, &bad), g->tgt = buf(g, N * 4, &bad), g->loss = buf(g, N * 4, &bad);
    g->xf = buf(g, ND, &bad), g->nf = buf(g, ND, &bad), g->fr = buf(g, N * 4, &bad);
    g->alpha = buf(g, N * c->V * cfb, &bad), g->galpha = buf(g, N * c->V * cfb, &bad);
    g->gx = buf(g, ND, &bad), g->gh = buf(g, ND, &bad), g->gq = buf(g, ND, &bad), g->gk = buf(g, ND, &bad);
    g->gv = buf(g, ND, &bad), g->gatt = buf(g, ND, &bad), g->gtmp = buf(g, ND, &bad);
    g->gO = buf(g, att * cfb, &bad), g->pg = buf(g, N * pgM * 4, &bad), g->part = buf(g, CHUNKS * 4, &bad);
    for (int l = 0; l < c->L; l++) {
        Layer *y = &g->ly[l];
        y->x = buf(g, ND, &bad), y->n = buf(g, ND, &bad), y->xr = buf(g, N * 4, &bad);
        y->q = buf(g, ND, &bad), y->k = buf(g, ND, &bad), y->v = buf(g, ND, &bad);
        y->A = buf(g, att * 4, &bad), y->Z = buf(g, (size_t)B * c->H * c->T * 4, &bad), y->att = buf(g, ND, &bad);
        y->h = buf(g, ND, &bad), y->hr = buf(g, N * 4, &bad), y->cs = buf(g, (J + 2) * ND, &bad);
        y->m = buf(g, ND, &bad);
    }
    if (bad || !g->host) {
        release_buffers(g);
        snprintf(g->err, sizeof g->err, "not enough GPU memory for this brain and batch - lower Batch or Width");
        return -1;
    }
    if (upload_doubles(g, g->P, m->p, m->n) || put(g, g->decay, m->decay, m->n)) return -1;
    if (s && s->m) {
        if (upload_doubles(g, g->Mm, s->m, m->n) || upload_doubles(g, g->Vv, s->v, m->n)) return -1;
    } else if (run(g, K_FILL, m->n, "mi", g->Mm, (int)m->n) || run(g, K_FILL, m->n, "mi", g->Vv, (int)m->n))
        return -1;
    cl.Finish(g->q);
    return 0;
}

/* Forward pass over the bound batch; leaves per-row losses in g->loss. */
static int forward(Gpu *g, const Model *m, const int *X, const int *Y, int grad)
{
    const Config *c = &g->c;
    int B = g->B, T = c->T, D = c->D, H = c->H, V = c->V, J = 2 * c->R, N = B * T;
    if (put(g, g->tok, X, N * sizeof(int)) || put(g, g->tgt, Y, N * sizeof(int))) return -1;
    if (run(g, K_EMBED, (size_t)N * D, "mmimiii", g->tok, g->P, (int)m->emb, g->ly[0].x, T, D, N * D)) return -1;
    for (int l = 0; l < c->L; l++) {
        Layer *y = &g->ly[l];
        const BlockOff *b = &m->blk[l];
        cl_mem out = l + 1 < c->L ? g->ly[l + 1].x : g->xf;
        if (run(g, K_QNORM, N, "mmmii", y->x, y->n, y->xr, D, N) ||
            run(g, K_MM, (size_t)N * D, "mimmmiiii", g->P, (int)b->wq, y->n, y->q, y->n, D, D, N, 0) ||
            run(g, K_MM, (size_t)N * D, "mimmmiiii", g->P, (int)b->wk, y->n, y->k, y->n, D, D, N, 0) ||
            run(g, K_MM, (size_t)N * D, "mimmmiiii", g->P, (int)b->wv, y->n, y->v, y->n, D, D, N, 0) ||
            run(g, K_ATTN, (size_t)B * H * T, "mmmmmmiiii", y->q, y->k, y->v, y->A, y->Z, y->att, B, T, H, D) ||
            run(g, K_MM, (size_t)N * D, "mimmmiiii", g->P, (int)b->wo, y->att, y->h, y->x, D, D, N, 1) ||
            run(g, K_FF, N, "mmmmmiiiiiii", g->P, y->h, y->hr, y->cs, y->m, (int)b->theta, (int)b->phi, (int)b->psi,
                (int)b->bias, D, J, N) ||
            run(g, K_MM, (size_t)N * D, "mimmmiiii", g->P, (int)b->wf, y->m, out, y->h, D, D, N, 1))
            return -1;
    }
    if (run(g, K_QNORM, N, "mmmii", g->xf, g->nf, g->fr, D, N) ||
        run(g, K_MM, (size_t)N * V, "mimmmiiii", g->P, (int)m->wout, g->nf, g->alpha, g->nf, V, D, N, 0) ||
        run(g, K_BORN, N, "mmmmiifi", g->alpha, g->tgt, g->loss, g->galpha, V, N, 1.0 / N, grad))
        return -1;
    return 0;
}

static int backward(Gpu *g, const Model *m)
{
    const Config *c = &g->c;
    int B = g->B, T = c->T, D = c->D, H = c->H, V = c->V, J = 2 * c->R, N = B * T, half = D / 2;
    int pgM = 2 * J * half + 2 * D;
    if (run(g, K_FILL, g->n, "mi", g->G, (int)g->n) ||
        run(g, K_MM_BW, (size_t)V * D, "mmmiiii", g->galpha, g->nf, g->G, (int)m->wout, V, D, N) ||
        run(g, K_MM_BX, (size_t)N * D, "mimmiiii", g->P, (int)m->wout, g->galpha, g->gtmp, V, D, N, 0) ||
        run(g, K_QNORM_BACK, N, "mmmmiii", g->xf, g->fr, g->gtmp, g->gx, D, N, 0))
        return -1;
    for (int l = c->L - 1; l >= 0; l--) {
        Layer *y = &g->ly[l];
        const BlockOff *b = &m->blk[l];
        size_t ND = (size_t)N * D;
        if (run(g, K_COPY, ND, "mmi", g->gx, g->gh, (int)ND) ||
            run(g, K_MM_BW, (size_t)D * D, "mmmiiii", g->gx, y->m, g->G, (int)b->wf, D, D, N) ||
            run(g, K_MM_BX, ND, "mimmiiii", g->P, (int)b->wf, g->gx, g->gtmp, D, D, N, 0) ||
            run(g, K_FF_BACK, N, "mmmmmmmiiiiiii", g->P, y->h, y->hr, y->cs, g->gtmp, g->gh, g->pg, (int)b->theta,
                (int)b->phi, (int)b->psi, (int)b->bias, D, J, N) ||
            run(g, K_REDUCE, (size_t)J * half, "miiimii", g->pg, pgM, 0, J * half, g->G, (int)b->theta, N) ||
            run(g, K_REDUCE, (size_t)J * half, "miiimii", g->pg, pgM, J * half, J * half, g->G, (int)b->phi, N) ||
            run(g, K_REDUCE, D, "miiimii", g->pg, pgM, 2 * J * half, D, g->G, (int)b->psi, N) ||
            run(g, K_REDUCE, D, "miiimii", g->pg, pgM, 2 * J * half + D, D, g->G, (int)b->bias, N) ||
            run(g, K_COPY, ND, "mmi", g->gh, g->gx, (int)ND) ||
            run(g, K_MM_BW, (size_t)D * D, "mmmiiii", g->gh, y->att, g->G, (int)b->wo, D, D, N) ||
            run(g, K_MM_BX, ND, "mimmiiii", g->P, (int)b->wo, g->gh, g->gatt, D, D, N, 0) ||
            run(g, K_ATTN_BQ, (size_t)B * H * T, "mmmmmmmmiiii", y->q, y->k, y->v, y->A, y->Z, g->gatt, g->gO, g->gq, B,
                T, H, D) ||
            run(g, K_ATTN_BKV, (size_t)B * H * T, "mmmmmmiiii", y->q, y->A, g->gatt, g->gO, g->gk, g->gv, B, T, H, D) ||
            run(g, K_MM_BW, (size_t)D * D, "mmmiiii", g->gq, y->n, g->G, (int)b->wq, D, D, N) ||
            run(g, K_MM_BW, (size_t)D * D, "mmmiiii", g->gk, y->n, g->G, (int)b->wk, D, D, N) ||
            run(g, K_MM_BW, (size_t)D * D, "mmmiiii", g->gv, y->n, g->G, (int)b->wv, D, D, N) ||
            run(g, K_MM_BX, ND, "mimmiiii", g->P, (int)b->wq, g->gq, g->gtmp, D, D, N, 0) ||
            run(g, K_MM_BX, ND, "mimmiiii", g->P, (int)b->wk, g->gk, g->gtmp, D, D, N, 1) ||
            run(g, K_MM_BX, ND, "mimmiiii", g->P, (int)b->wv, g->gv, g->gtmp, D, D, N, 1) ||
            run(g, K_QNORM_BACK, N, "mmmmiii", y->x, y->xr, g->gtmp, g->gx, D, N, 1))
            return -1;
    }
    return run(g, K_EMBED_BACK, (size_t)V * D, "mmmiiiii", g->tok, g->gx, g->G, (int)m->emb, N, T, D, V);
}

static double mean_loss(Gpu *g)
{
    int N = g->B * g->c.T;
    if (get(g, g->loss, g->host, N * sizeof(float))) return NAN;
    double s = 0;
    for (int i = 0; i < N; i++) s += g->host[i];
    return s / N;
}

double gpu_step(Gpu *g, const Model *m, OptState *s, const int *X, const int *Y, const OptCfg *o, double lr,
                double *gnorm, uint64_t *rng)
{
    if (forward(g, m, X, Y, 1) || backward(g, m)) return NAN;
    double loss = mean_loss(g);
    float part[CHUNKS];
    if (run(g, K_SUMSQ, CHUNKS, "mimi", g->G, (int)g->n, g->part, CHUNKS) || get(g, g->part, part, sizeof part))
        return NAN;
    double ss = 0;
    for (int i = 0; i < CHUNKS; i++) ss += part[i];
    double norm = sqrt(ss), gs = o->clip > 0 && norm > o->clip ? o->clip / norm : 1;
    if (gnorm) *gnorm = norm;
    s->t++;
    double c1 = 1 - pow(o->beta1, s->t), c2 = 1 - pow(o->beta2, s->t);
    if (run(g, K_OPT, g->n, "mmmmmiiffffffffu", g->P, g->G, g->Mm, g->Vv, g->decay, (int)g->n, o->type, lr, o->beta1,
            o->beta2, c1, c2, o->wd, gs, o->anneal * lr, (unsigned)(rng_u64(rng) >> 32)))
        return NAN;
    return loss;
}

double gpu_eval(Gpu *g, const Model *m, const int *X, const int *Y)
{
    if (forward(g, m, X, Y, 0)) return NAN;
    return mean_loss(g);
}

static int download_doubles(Gpu *g, cl_mem src, double *dst)
{
    if (get(g, src, g->host, g->n * sizeof(float))) return -1;
    for (size_t i = 0; i < g->n; i++) dst[i] = g->host[i];
    return 0;
}

int gpu_pull(Gpu *g, Model *m, OptState *s)
{
    if (!g->B || m->n != g->n) return -1;
    if (download_doubles(g, g->P, m->p)) return -1;
    if (s && s->m && (download_doubles(g, g->Mm, s->m) || download_doubles(g, g->Vv, s->v))) return -1;
    return 0;
}

int gpu_grad(Gpu *g, const Model *m, const int *X, const int *Y, double *grad, double *loss)
{
    if (forward(g, m, X, Y, 1) || backward(g, m)) return -1;
    *loss = mean_loss(g);
    return download_doubles(g, g->G, grad);
}
