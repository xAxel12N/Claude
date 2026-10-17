/* main.c - QuantumGPT command line: train (with live start/stop controls), sample, gradcheck. */
#include "qcore.h"
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    const char *mode, *data, *ckpt, *prompt;
    Config c;
    OptCfg o;
    int batch, threads, eval_every, len, fresh, autostart;
    double temp;
    uint64_t seed;
} Args;

/* ---------- controls: keyboard "buttons" and signals from Quantom.sh ---------- */

static volatile sig_atomic_t g_quit, g_pause, g_resume;
static void on_quit(int s) { (void)s, g_quit = 1; }
static void on_pause(int s) { (void)s, g_pause = 1; }
static void on_resume(int s) { (void)s, g_resume = 1; }

static struct termios g_tio;
static int g_tty;

static void tty_restore(void)
{
    if (g_tty) tcsetattr(0, TCSANOW, &g_tio);
}

static void tty_raw(void)
{
    if (!isatty(0) || !isatty(1) || tcgetattr(0, &g_tio)) return;
    struct termios t = g_tio;
    t.c_lflag &= ~(ICANON | ECHO); /* keys arrive immediately; Ctrl-C still works */
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    tcsetattr(0, TCSANOW, &t);
    g_tty = 1;
    atexit(tty_restore);
}

static int key(void)
{
    unsigned char ch;
    return g_tty && read(0, &ch, 1) == 1 ? ch : -1;
}

static void install_signals(void)
{
    struct sigaction sa = {0};
    sa.sa_handler = on_quit;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sa.sa_handler = on_pause;
    sigaction(SIGUSR1, &sa, NULL);
    sa.sa_handler = on_resume;
    sigaction(SIGUSR2, &sa, NULL);
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void nap(int ms)
{
    struct timespec ts = {0, ms * 1000000L};
    nanosleep(&ts, NULL);
}

/* ---------- training ---------- */

static void generate(const Model *m, Work *w, const Data *d, const char *prompt, int len, double temp,
                     uint64_t *rng)
{
    int T = m->c.T, cap = (int)strlen(prompt) + len + 1;
    int *buf = malloc(sizeof(int) * cap), n = data_encode(d, prompt, buf, cap);
    if (n == 0) buf[n++] = d->stoi['\n'] >= 0 ? d->stoi['\n'] : 0;
    for (int i = 0; i < n; i++) putchar(d->itos[buf[i]]);
    for (int i = 0; i < len; i++) {
        int start = n > T ? n - T : 0;
        buf[n] = model_sample(m, w, buf + start, n - start, temp, rng);
        putchar(d->itos[buf[n++]]);
        fflush(stdout);
    }
    putchar('\n');
    free(buf);
}

static const char *HELP_KEYS = "  [SPACE] start/stop   [g] sample   [s] save   [+/-] learning rate   [q] save & quit";

static void status(int paused, long step, long steps, double loss, double val, double best, double lr, double gn,
                   double tps)
{
    char vs[48] = "val -";
    if (isfinite(val)) snprintf(vs, sizeof vs, "val %.3f (best %.3f)", val, best);
    if (g_tty) printf("\r\033[K");
    printf("%s step %ld/%ld | loss %.3f | %s | lr %.2e | |g| %.2f | %.0f tok/s", paused ? "[STOPPED]" : "[RUNNING]",
           step, steps, loss, vs, lr, gn, tps);
    if (!g_tty) putchar('\n');
    fflush(stdout);
}

static int run_train(Args *a)
{
    Data d;
    Model m;
    OptState s;
    long step = 0;
    double best = INFINITY;
    uint64_t rng = a->seed;
    if (data_load(&d, a->data, a->c.T)) return 1;

    if (!a->fresh && access(a->ckpt, F_OK) == 0) {
        Data cv;
        if (ckpt_load(a->ckpt, &m, &s, &cv, &step, &best)) return 1;
        if (cv.V != d.V || memcmp(cv.itos, d.itos, d.V)) {
            fprintf(stderr, "%s was trained on different text; use --fresh to start over\n", a->ckpt);
            return 1;
        }
        printf("resumed %s at step %ld\n", a->ckpt, step);
    } else {
        a->c.V = d.V;
        if (model_init(&m, a->c, &rng) || opt_init(&s, m.n)) return 1;
    }
    Config c = m.c;
    int T = c.T, B = a->batch, nth = learn_threads(a->threads);
    if (a->o.min_lr < 0) a->o.min_lr = a->o.lr * 0.1;

    printf("QuantumGPT | %zu real parameters | vocab %d | width %d complex | heads %d | blocks %d | context %d | "
           "circuit depth %d\n",
           m.n, c.V, c.D, c.H, c.L, c.T, c.R);
    printf("learning: %s, lr %.1e (%s, warmup %ld), weight decay %.3g, clip %.2g, anneal %.3g | batch %d | %d "
           "thread(s)\n",
           opt_name(a->o.type), a->o.lr, sched_name(a->o.sched), a->o.warmup, a->o.wd, a->o.clip, a->o.anneal, B,
           nth);
    printf("text: %ld chars (%ld train / %ld validation)\n", d.n, d.split, d.n - d.split);

    Work **ws = malloc(sizeof *ws * nth);
    for (int i = 0; i < nth; i++) ws[i] = work_new(&m);
    int *X = malloc(sizeof(int) * B * T), *Y = malloc(sizeof(int) * B * T);

    install_signals();
    tty_raw();
    int paused = g_tty && !a->autostart; /* interactive runs wait for the start button */
    if (g_tty) printf("%s\n", HELP_KEYS);
    if (paused) printf("press SPACE to start training\n");

    double loss = NAN, val = NAN, gn = 0, tps = 0, t0 = now();
    long tok_count = 0;
    status(paused, step, a->o.steps, loss, val, best, opt_lr(&a->o, step), gn, tps);
    while (!g_quit && step < a->o.steps) {
        int k, want_sample = 0, want_save = 0;
        while ((k = key()) >= 0) {
            if (k == ' ' || k == 'p') paused = !paused;
            else if (k == 'g') want_sample = 1;
            else if (k == 's' || k == 'w') want_save = 1;
            else if (k == 'q' || k == 27) g_quit = 1;
            else if (k == '+' || k == '=') a->o.lr *= 1.5, a->o.min_lr *= 1.5;
            else if (k == '-' || k == '_') a->o.lr /= 1.5, a->o.min_lr /= 1.5;
            status(paused, step, a->o.steps, loss, val, best, opt_lr(&a->o, step), gn, tps);
        }
        if (g_pause) g_pause = 0, paused = 1, status(1, step, a->o.steps, loss, val, best, opt_lr(&a->o, step), gn, tps);
        if (g_resume) g_resume = 0, paused = 0;
        if (want_sample) {
            printf("\n--- sample ---\n");
            generate(&m, ws[0], &d, "", 300, a->temp, &rng);
            printf("--------------\n");
        }
        if (want_save && !ckpt_save(a->ckpt, &m, &s, &d, step, best)) printf("\nsaved %s\n", a->ckpt);
        if (g_quit) break;
        if (paused) {
            nap(50);
            t0 = now(), tok_count = 0;
            continue;
        }

        for (int b = 0; b < B; b++) data_batch(&d, 0, T, X + b * T, Y + b * T, &rng);
        double lr = opt_lr(&a->o, step);
        loss = learn_batch(&m, ws, nth, X, Y, B, T);
        gn = opt_clip(work_grad(ws[0]), m.n, a->o.clip);
        opt_step(&a->o, &s, &m, work_grad(ws[0]), lr, &rng);
        step++;
        tok_count += (long)B * T;
        tps = tok_count / (now() - t0 + 1e-9);
        if (!isfinite(loss)) {
            fprintf(stderr, "\nloss became %f - try a smaller --lr\n", loss);
            break;
        }

        int last = step == a->o.steps;
        if (step % a->eval_every == 0 || last) {
            val = learn_eval(&m, ws, nth, &d, 4 * B, &rng);
            if (val < best) {
                best = val;
                char path[1024];
                snprintf(path, sizeof path, "%s.best", a->ckpt);
                ckpt_save(path, &m, &s, &d, step, best);
            }
            ckpt_save(a->ckpt, &m, &s, &d, step, best);
            status(paused, step, a->o.steps, loss, val, best, lr, gn, tps);
            printf("\n");
            generate(&m, ws[0], &d, "", 160, a->temp, &rng);
        } else if (g_tty || step % 10 == 0) {
            status(paused, step, a->o.steps, loss, val, best, lr, gn, tps);
        }
    }
    ckpt_save(a->ckpt, &m, &s, &d, step, best);
    printf("\nstopped at step %ld, saved %s (best validation loss %.3f)\n", step, a->ckpt, best);

    for (int i = 0; i < nth; i++) work_free(ws[i]);
    free(ws);
    free(X);
    free(Y);
    opt_free(&s);
    model_free(&m);
    data_free(&d);
    return 0;
}

static int run_sample(Args *a)
{
    Model m;
    Data d;
    long step;
    double best;
    uint64_t rng = a->seed;
    if (ckpt_load(a->ckpt, &m, NULL, &d, &step, &best)) return 1;
    fprintf(stderr, "[%s: step %ld, best validation loss %.3f]\n", a->ckpt, step, best);
    Work *w = work_new(&m);
    generate(&m, w, &d, a->prompt, a->len, a->temp, &rng);
    work_free(w);
    model_free(&m);
    return 0;
}

/* Compares every hand-written gradient with a finite-difference estimate on a tiny model. */
static int run_gradcheck(uint64_t seed)
{
    Config c = {.V = 11, .D = 8, .H = 2, .L = 2, .T = 7, .R = 1};
    Model m;
    uint64_t rng = seed;
    if (model_init(&m, c, &rng)) return 1;
    Work *w = work_new(&m);
    int tok[7], tgt[7];
    for (int i = 0; i < c.T; i++) tok[i] = rng_u64(&rng) % c.V, tgt[i] = rng_u64(&rng) % c.V;
    model_forward(&m, w, tok, tgt, c.T);
    model_backward(&m, w, tok, tgt, c.T, 1.0);
    const double *g = work_grad(w), h = 1e-5;
    double worst = 0;
    size_t worst_i = 0;
    for (size_t i = 0; i < m.n; i++) {
        double old = m.p[i];
        m.p[i] = old + h;
        double lp = model_forward(&m, w, tok, tgt, c.T);
        m.p[i] = old - h;
        double lm = model_forward(&m, w, tok, tgt, c.T);
        m.p[i] = old;
        double num = (lp - lm) / (2 * h), err = fabs(num - g[i]) / fmax(1e-6, fabs(num) + fabs(g[i]));
        if (err > worst) worst = err, worst_i = i;
    }
    int ok = worst < 1e-4;
    printf("gradcheck: %zu parameters, worst relative error %.2e (param %zu) -> %s\n", m.n, worst, worst_i,
           ok ? "PASS" : "FAIL");
    work_free(w);
    model_free(&m);
    return !ok;
}

static void usage(void)
{
    puts("usage: quantum <train|sample|gradcheck> [options]\n"
         "  --data FILE      training text (data/input.txt)\n"
         "  --ckpt FILE      checkpoint (quantum.ckpt); training resumes from it\n"
         "  --fresh          ignore an existing checkpoint\n"
         "  --start          start training at once instead of waiting for SPACE\n"
         "model:    --width 64 --heads 4 --layers 2 --ctx 64 --circuit 2\n"
         "learning: --steps 3000 --batch 16 --lr 3e-3 --min-lr X --opt sgd|momentum|adam|adamw\n"
         "          --sched const|cosine|step --warmup 100 --wd 0.01 --clip 1 --anneal 0\n"
         "          --beta1 0.9 --beta2 0.99 --eval-every 250 --threads N --seed 1\n"
         "sampling: --prompt TEXT --len 500 --temp 0.8");
}

int main(int argc, char **argv)
{
    Args a = {.data = "data/input.txt", .ckpt = "quantum.ckpt", .prompt = "", .batch = 16, .eval_every = 250,
              .len = 500, .temp = 0.8, .seed = 1};
    a.c = (Config){.D = 64, .H = 4, .L = 2, .T = 64, .R = 2};
    a.o = (OptCfg){.type = OPT_ADAMW, .sched = SCHED_COSINE, .lr = 3e-3, .min_lr = -1, .beta1 = 0.9, .beta2 = 0.99,
                   .wd = 0.01, .clip = 1.0, .warmup = 100, .steps = 3000};
    for (int i = 1; i < argc; i++) {
        const char *k = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
#define OPT(name) (!strcmp(k, name) && v && ++i)
        if (!strcmp(k, "-h") || !strcmp(k, "--help")) return usage(), 0;
        else if (!strcmp(k, "--fresh")) a.fresh = 1;
        else if (!strcmp(k, "--start")) a.autostart = 1;
        else if (OPT("--data")) a.data = v;
        else if (OPT("--ckpt")) a.ckpt = v;
        else if (OPT("--prompt")) a.prompt = v;
        else if (OPT("--width")) a.c.D = atoi(v);
        else if (OPT("--heads")) a.c.H = atoi(v);
        else if (OPT("--layers")) a.c.L = atoi(v);
        else if (OPT("--ctx")) a.c.T = atoi(v);
        else if (OPT("--circuit")) a.c.R = atoi(v);
        else if (OPT("--steps")) a.o.steps = atol(v);
        else if (OPT("--batch")) a.batch = atoi(v);
        else if (OPT("--lr")) a.o.lr = atof(v);
        else if (OPT("--min-lr")) a.o.min_lr = atof(v);
        else if (OPT("--warmup")) a.o.warmup = atol(v);
        else if (OPT("--wd")) a.o.wd = atof(v);
        else if (OPT("--clip")) a.o.clip = atof(v);
        else if (OPT("--anneal")) a.o.anneal = atof(v);
        else if (OPT("--beta1")) a.o.beta1 = atof(v);
        else if (OPT("--beta2")) a.o.beta2 = atof(v);
        else if (OPT("--eval-every")) a.eval_every = atoi(v);
        else if (OPT("--threads")) a.threads = atoi(v);
        else if (OPT("--seed")) a.seed = strtoull(v, NULL, 10);
        else if (OPT("--len")) a.len = atoi(v);
        else if (OPT("--temp")) a.temp = atof(v);
        else if (OPT("--opt")) {
            if ((a.o.type = opt_parse(v)) < 0) return fprintf(stderr, "unknown optimiser %s\n", v), 2;
        } else if (OPT("--sched")) {
            if ((a.o.sched = sched_parse(v)) < 0) return fprintf(stderr, "unknown schedule %s\n", v), 2;
        } else if (k[0] != '-' && !a.mode)
            a.mode = k;
        else
            return fprintf(stderr, "unknown or incomplete option %s\n", k), usage(), 2;
#undef OPT
    }
    if (a.seed == 0) a.seed = 1;
    if (a.batch < 1 || a.eval_every < 1 || a.o.steps < 1) return fprintf(stderr, "batch, steps, eval-every must be > 0\n"), 2;
    if (!a.mode || !strcmp(a.mode, "train")) return run_train(&a);
    if (!strcmp(a.mode, "sample")) return run_sample(&a);
    if (!strcmp(a.mode, "gradcheck")) return run_gradcheck(a.seed);
    usage();
    return 2;
}
