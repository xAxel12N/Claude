/* gui.c - QuantumGPT window: train on your .txt files, watch the brain think, chat with it,
 * and keep an eye on temperatures and fans. Built with raylib. */
#define Model RlModel /* raylib has its own Model type */
#include "raylib.h"
#undef Model
#include "qcore.h"
#include "gpu.h"
#include "sensors.h"
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define TEXT_DIR "texts"
#define CKPT "quantum.ckpt"
#define MAXF 48
#define HMAX 1500
#define MAXMSG 60
#define MSGLEN 1000

static const Color BG = {11, 16, 32, 255}, PANEL = {18, 25, 48, 255}, EDGE = {40, 51, 92, 255},
                   ACC = {124, 196, 255, 255}, FG = {232, 236, 248, 255}, DIM = {138, 147, 178, 255},
                   GOOD = {94, 224, 122, 255}, BAD = {255, 107, 107, 255}, WARN = {255, 200, 87, 255};

typedef struct {
    char name[200];
    long size;
    int on;
} TextFile;

typedef struct {
    int ai;
    char text[MSGLEN];
} Msg;

/* Everything the window, the trainer thread and the chat thread share.
 * mu guards the small fields; model_mu guards the model, optimiser and data. */
static struct {
    pthread_mutex_t mu, model_mu;
    /* requests from the window */
    int running, quit, want_save, want_reload, want_reset, want_inspect;
    Config new_cfg;
    OptCfg o;
    int batch, eval_every;
    /* the brain (model_mu) */
    Model m;
    OptState s;
    Data voc, d;
    int have_model, have_data, brain_id, nth;
    Work **ws, *gw;
    /* progress (mu) */
    long step;
    double loss, val, best, tps;
    float hl[HMAX], vl[HMAX];
    long hs[HMAX], vs[HMAX];
    int nh, nv, hstride, hcount;
    double hacc;
    char sample[400], note[220];
    double note_t;
    Inspect ins;
    char probe[128];
    TextFile files[MAXF];
    int nfiles;
    Msg chat[MAXMSG];
    int nchat, generating;
    double creativity;
    int use_gpu, gpu_ok;
    char dev_label[200];
    Config cur; /* shape of the current brain, for display */
} A;

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void nap(int ms)
{
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

static void note(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    pthread_mutex_lock(&A.mu);
    vsnprintf(A.note, sizeof A.note, fmt, ap);
    A.note_t = now();
    pthread_mutex_unlock(&A.mu);
    va_end(ap);
}

/* ---------- training texts ---------- */

static int is_text(const char *name)
{
    const char *dot = strrchr(name, '.');
    return name[0] != '.' && dot && (!strcasecmp(dot, ".txt") || !strcasecmp(dot, ".md"));
}

static int was_disabled(const char *name)
{
    char line[256];
    int found = 0;
    FILE *f = fopen(TEXT_DIR "/.disabled", "r");
    if (!f) return 0;
    while (!found && fgets(line, sizeof line, f)) found = !strcmp(strtok(line, "\n") ? line : "", name);
    fclose(f);
    return found;
}

static void save_disabled(void)
{
    FILE *f = fopen(TEXT_DIR "/.disabled", "w");
    if (!f) return;
    for (int i = 0; i < A.nfiles; i++)
        if (!A.files[i].on) fprintf(f, "%s\n", A.files[i].name);
    fclose(f);
}

static int cmp_files(const void *a, const void *b) { return strcmp(((const TextFile *)a)->name, ((const TextFile *)b)->name); }

/* Rescans the texts folder; asks the trainer to reload when the enabled set changed. */
static void scan_files(void)
{
    TextFile nf[MAXF];
    int n = 0;
    DIR *dir = opendir(TEXT_DIR);
    if (dir) {
        struct dirent *e;
        while ((e = readdir(dir)) && n < MAXF) {
            if (!is_text(e->d_name) || strlen(e->d_name) >= sizeof nf[0].name) continue;
            char path[300];
            struct stat st;
            snprintf(path, sizeof path, TEXT_DIR "/%s", e->d_name);
            if (stat(path, &st) || !S_ISREG(st.st_mode)) continue;
            snprintf(nf[n].name, sizeof nf[n].name, "%s", e->d_name);
            nf[n].size = st.st_size;
            nf[n].on = -1;
            n++;
        }
        closedir(dir);
    }
    qsort(nf, n, sizeof *nf, cmp_files);
    pthread_mutex_lock(&A.mu);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < A.nfiles; j++)
            if (!strcmp(nf[i].name, A.files[j].name)) nf[i].on = A.files[j].on;
    for (int i = 0; i < n; i++)
        if (nf[i].on < 0) nf[i].on = !was_disabled(nf[i].name);
    int changed = n != A.nfiles;
    for (int i = 0; !changed && i < n; i++)
        changed = strcmp(nf[i].name, A.files[i].name) || nf[i].size != A.files[i].size || nf[i].on != A.files[i].on;
    memcpy(A.files, nf, sizeof(TextFile) * n);
    A.nfiles = n;
    if (changed) A.want_reload = 1;
    pthread_mutex_unlock(&A.mu);
}

static void import_file(const char *src)
{
    const char *base = strrchr(src, '/');
    base = base ? base + 1 : src;
    char dst[400], stem[200];
    snprintf(stem, sizeof stem, "%s", base);
    char *dot = strrchr(stem, '.');
    int has_ext = dot && is_text(base);
    if (has_ext) *dot = 0;
    snprintf(dst, sizeof dst, TEXT_DIR "/%s.txt", stem);
    for (int i = 2; access(dst, F_OK) == 0 && i < 100; i++) snprintf(dst, sizeof dst, TEXT_DIR "/%s-%d.txt", stem, i);
    char buf[65536];
    size_t k;
    FILE *in = fopen(src, "rb");
    if (in) { /* PDFs, images and the like contain zero bytes; plain text does not */
        k = fread(buf, 1, 4096, in);
        rewind(in);
        if (memchr(buf, 0, k)) {
            fclose(in);
            note("%s is not a plain text file - save it as .txt first", base);
            return;
        }
    }
    FILE *out = in ? fopen(dst, "wb") : NULL;
    if (!in || !out) {
        if (in) fclose(in);
        note("Could not copy %s", base);
        return;
    }
    while ((k = fread(buf, 1, sizeof buf, in)) > 0) fwrite(buf, 1, k, out);
    fclose(in);
    fclose(out);
    note("Added %s - it is used from the next training step", dst + sizeof TEXT_DIR);
}

static void *pick_files(void *arg)
{
    (void)arg;
    if (system("command -v zenity >/dev/null 2>&1")) {
        note("No file picker (zenity) found - drag .txt files onto the window instead");
        return NULL;
    }
    FILE *p = popen("zenity --file-selection --multiple --title='Choose text files to learn from' "
                    "--file-filter='Text files | *.txt *.md' --file-filter='All files | *' --separator='\n' 2>/dev/null",
                    "r");
    if (!p) return NULL;
    char line[1024];
    while (fgets(line, sizeof line, p)) {
        line[strcspn(line, "\n")] = 0;
        if (line[0]) import_file(line);
    }
    pclose(p);
    return NULL;
}

/* ---------- the trainer thread ---------- */

static void hist_push(double loss, long step)
{
    A.hacc += loss;
    if (++A.hcount < A.hstride) return;
    if (A.nh == HMAX) { /* full: halve the resolution */
        for (int i = 0; i < HMAX / 2; i++) A.hl[i] = (A.hl[2 * i] + A.hl[2 * i + 1]) / 2, A.hs[i] = A.hs[2 * i + 1];
        A.nh = HMAX / 2;
        A.hstride *= 2;
    }
    A.hl[A.nh] = A.hacc / A.hcount;
    A.hs[A.nh++] = step;
    A.hacc = 0, A.hcount = 0;
}

static void val_push(double val, long step)
{
    if (A.nv == HMAX) {
        memmove(A.vl, A.vl + HMAX / 2, sizeof(float) * HMAX / 2);
        memmove(A.vs, A.vs + HMAX / 2, sizeof(long) * HMAX / 2);
        A.nv = HMAX / 2;
    }
    A.vl[A.nv] = val;
    A.vs[A.nv++] = step;
}

static void free_brain(void)
{
    for (int i = 0; i < A.nth; i++) work_free(A.ws[i]);
    free(A.ws);
    work_free(A.gw);
    A.ws = NULL, A.gw = NULL;
    opt_free(&A.s);
    model_free(&A.m);
    A.have_model = 0;
}

static void alloc_works(void)
{
    A.nth = learn_threads(0);
    A.ws = malloc(sizeof *A.ws * A.nth);
    for (int i = 0; i < A.nth; i++) A.ws[i] = work_new(&A.m);
    A.gw = work_new(&A.m);
}

static void clear_history(void)
{
    pthread_mutex_lock(&A.mu);
    A.nh = A.nv = A.hcount = 0;
    A.hstride = 1;
    A.hacc = 0;
    A.loss = A.val = NAN;
    A.sample[0] = 0;
    pthread_mutex_unlock(&A.mu);
}

/* Loads the enabled texts. A fresh brain (or none yet) learns its alphabet from them. */
static void reload(int fresh, uint64_t *rng)
{
    const char *paths[MAXF];
    char store[MAXF][300];
    int np = 0;
    pthread_mutex_lock(&A.mu);
    for (int i = 0; i < A.nfiles; i++)
        if (A.files[i].on) snprintf(store[np], sizeof store[np], TEXT_DIR "/%s", A.files[i].name), paths[np] = store[np], np++;
    Config nc = A.new_cfg;
    pthread_mutex_unlock(&A.mu);

    pthread_mutex_lock(&A.model_mu);
    if (np == 0) {
        if (A.have_data) data_free(&A.d);
        A.have_data = 0;
        pthread_mutex_unlock(&A.model_mu);
        note("Add a .txt file (drag it onto the window) so the brain has something to learn");
        return;
    }
    Data nd;
    long dropped = 0;
    int make = fresh || !A.have_model;
    int T = make ? nc.T : A.m.c.T;
    if (data_load_files(&nd, paths, np, make ? NULL : &A.voc, T, &dropped)) {
        pthread_mutex_unlock(&A.model_mu);
        note("Those texts are too short to learn from - add more text");
        return;
    }
    if (make) {
        if (A.have_model) free_brain();
        nc.V = nd.V;
        if (model_init(&A.m, nc, rng) || opt_init(&A.s, A.m.n)) {
            data_free(&nd);
            pthread_mutex_unlock(&A.model_mu);
            note("Could not create that brain");
            return;
        }
        A.voc = nd;
        A.voc.tok = NULL;
        alloc_works();
        A.have_model = 1;
        A.brain_id++;
        pthread_mutex_lock(&A.mu);
        A.step = 0;
        A.best = INFINITY;
        A.cur = A.m.c;
        if (A.o.steps < 5000) A.o.steps = 5000;
        pthread_mutex_unlock(&A.mu);
        clear_history();
    }
    if (A.have_data) data_free(&A.d);
    A.d = nd;
    A.have_data = 1;
    A.want_inspect = 1;
    pthread_mutex_unlock(&A.model_mu);
    if (make)
        note("New brain: %d characters in its alphabet, %ld characters of text to learn from", A.voc.V, nd.n);
    else if (dropped)
        note("Learning from %d file(s), %ld characters. %ld unknown characters skipped - make a New brain to learn them",
             np, nd.n, dropped);
    else
        note("Learning from %d file(s), %ld characters", np, nd.n);
}

/* The graphics card, when training runs there (trainer thread only). */
static Gpu *g_gpu;
static int g_on_gpu, g_gpu_id = -1, g_gpu_B;

/* While the card holds the weights, refresh the CPU copy. Call with model_mu held. */
static void sync_from_gpu(int with_opt)
{
    if (g_on_gpu && g_gpu_id == A.brain_id) gpu_pull(g_gpu, &A.m, with_opt ? &A.s : NULL);
}

static void save_brain(void)
{
    pthread_mutex_lock(&A.model_mu);
    sync_from_gpu(1);
    int ok = A.have_model && !ckpt_save(CKPT, &A.m, &A.s, &A.voc, A.step, A.best);
    pthread_mutex_unlock(&A.model_mu);
    if (ok) note("Saved the brain to " CKPT);
}

static void inspect(void)
{
    char probe[128];
    pthread_mutex_lock(&A.mu);
    snprintf(probe, sizeof probe, "%s", A.probe);
    pthread_mutex_unlock(&A.mu);
    Inspect ins;
    int tok[128];
    pthread_mutex_lock(&A.model_mu);
    sync_from_gpu(0);
    int n = A.have_model ? data_encode(&A.voc, probe, tok, 128) : 0;
    if (n > 0) model_inspect(&A.m, A.gw, tok, n, A.voc.itos, &ins);
    pthread_mutex_unlock(&A.model_mu);
    if (n <= 0) memset(&ins, 0, sizeof ins);
    pthread_mutex_lock(&A.mu);
    A.ins = ins;
    pthread_mutex_unlock(&A.mu);
}

static void sample_text(uint64_t *rng)
{
    char out[200];
    int len = 0, T = A.m.c.T, buf[400], n = 1;
    buf[0] = A.voc.stoi['\n'] >= 0 ? A.voc.stoi['\n'] : 0;
    for (; len < 160 && n < 400; len++) {
        int start = n > T ? n - T : 0;
        buf[n] = model_sample(&A.m, A.gw, buf + start, n - start, 0.8, rng);
        out[len] = A.voc.itos[buf[n++]];
    }
    out[len] = 0;
    for (char *c = out; *c; c++)
        if (*c == '\n' || *c == '\r' || *c == '\t') *c = ' ';
    pthread_mutex_lock(&A.mu);
    snprintf(A.sample, sizeof A.sample, "%s", out);
    pthread_mutex_unlock(&A.mu);
}

static void *trainer(void *arg)
{
    (void)arg;
    uint64_t rng = (uint64_t)time(NULL) | 1;
    int *X = NULL, *Y = NULL, cap = 0;
    double last_inspect = 0;
    char gerr[256];
    g_gpu = gpu_open(gerr, sizeof gerr);
    pthread_mutex_lock(&A.mu);
    A.gpu_ok = g_gpu != NULL;
    snprintf(A.dev_label, sizeof A.dev_label, "%s", g_gpu ? gpu_name(g_gpu) : gerr);
    if (g_gpu) A.use_gpu = 1, A.batch = 64; /* a card likes big batches */
    pthread_mutex_unlock(&A.mu);

    if (access(CKPT, F_OK) == 0) {
        Data voc;
        long step;
        double best;
        if (ckpt_load(CKPT, &A.m, &A.s, &voc, &step, &best) == 0) {
            pthread_mutex_lock(&A.model_mu);
            A.voc = voc;
            alloc_works();
            A.have_model = 1;
            A.brain_id++;
            pthread_mutex_unlock(&A.model_mu);
            pthread_mutex_lock(&A.mu);
            A.step = step, A.best = best, A.cur = A.m.c;
            A.new_cfg = A.m.c;
            A.o.steps = (step / 1000 + 5) * 1000;
            pthread_mutex_unlock(&A.mu);
            note("Loaded your trained brain from " CKPT " (step %ld) - press START to keep teaching it", step);
        } else
            note(CKPT " could not be read - a new brain will be made");
    }

    for (;;) {
        pthread_mutex_lock(&A.mu);
        int quit = A.quit, running = A.running, reload_now = A.want_reload || A.want_reset, fresh = A.want_reset;
        int save = A.want_save, insp = A.want_inspect;
        A.want_reload = A.want_reset = A.want_save = A.want_inspect = 0;
        OptCfg o = A.o;
        int B = A.batch, eval_every = A.eval_every, use_gpu = A.use_gpu;
        long step = A.step;
        pthread_mutex_unlock(&A.mu);
        if (quit) break;
        if (reload_now) reload(fresh, &rng);
        pthread_mutex_lock(&A.model_mu);
        if (g_on_gpu && g_gpu_id != A.brain_id) g_on_gpu = 0; /* that brain was replaced */
        if (g_on_gpu && (!use_gpu || g_gpu_B != B)) gpu_pull(g_gpu, &A.m, &A.s), g_on_gpu = 0;
        pthread_mutex_unlock(&A.model_mu);
        if (save) save_brain();
        if (insp || (running && now() - last_inspect > 0.8)) inspect(), last_inspect = now();
        if (!running || !A.have_model || !A.have_data) {
            nap(30);
            continue;
        }
        if (step >= o.steps) {
            pthread_mutex_lock(&A.mu);
            A.running = 0;
            pthread_mutex_unlock(&A.mu);
            save_brain();
            note("Reached the step limit (%ld) - raise Steps on the left to keep going", o.steps);
            continue;
        }

        pthread_mutex_lock(&A.model_mu);
        int bind_failed = 0;
        if (use_gpu && g_gpu && !g_on_gpu) {
            if (gpu_bind(g_gpu, &A.m, &A.s, B) == 0) g_on_gpu = 1, g_gpu_id = A.brain_id, g_gpu_B = B;
            else bind_failed = 1;
        }
        pthread_mutex_unlock(&A.model_mu);
        if (bind_failed) {
            pthread_mutex_lock(&A.mu);
            A.use_gpu = 0;
            pthread_mutex_unlock(&A.mu);
            note("GPU: %s - training on the CPU instead", gpu_error(g_gpu));
            continue;
        }

        double t0 = now();
        pthread_mutex_lock(&A.model_mu);
        int T = A.m.c.T;
        if (B * T > cap) {
            cap = B * T;
            X = realloc(X, sizeof(int) * cap);
            Y = realloc(Y, sizeof(int) * cap);
        }
        for (int b = 0; b < B; b++) data_batch(&A.d, 0, T, X + b * T, Y + b * T, &rng);
        double lr = opt_lr(&o, step), loss;
        int gpu_failed = 0;
        if (g_on_gpu) {
            loss = gpu_step(g_gpu, &A.m, &A.s, X, Y, &o, lr, NULL, &rng);
            gpu_failed = !isfinite(loss) && gpu_error(g_gpu)[0];
        } else {
            loss = learn_batch(&A.m, A.ws, A.nth, X, Y, B, T);
            opt_clip(work_grad(A.ws[0]), A.m.n, o.clip);
            opt_step(&o, &A.s, &A.m, work_grad(A.ws[0]), lr, &rng);
        }
        step++;
        double val = NAN;
        if (step % eval_every == 0 && !gpu_failed) {
            if (g_on_gpu) {
                val = 0;
                for (int e = 0; e < 4; e++) {
                    for (int b = 0; b < B; b++) data_batch(&A.d, 1, T, X + b * T, Y + b * T, &rng);
                    val += gpu_eval(g_gpu, &A.m, X, Y) / 4;
                }
                gpu_pull(g_gpu, &A.m, &A.s);
            } else
                val = learn_eval(&A.m, A.ws, A.nth, &A.d, 4 * B, &rng);
            sample_text(&rng);
        }
        if (gpu_failed) g_on_gpu = 0;
        pthread_mutex_unlock(&A.model_mu);
        if (gpu_failed) {
            pthread_mutex_lock(&A.mu);
            A.use_gpu = 0, A.running = 0;
            pthread_mutex_unlock(&A.mu);
            note("GPU: %s - switched to the CPU, press START to go on", gpu_error(g_gpu));
            continue;
        }

        pthread_mutex_lock(&A.mu);
        double dt = now() - t0, tps = B * T / (dt + 1e-9);
        A.tps = A.tps > 0 ? 0.9 * A.tps + 0.1 * tps : tps;
        A.step = step;
        A.loss = isfinite(A.loss) ? 0.95 * A.loss + 0.05 * loss : loss;
        hist_push(loss, step);
        int better = 0;
        if (isfinite(val)) {
            A.val = val;
            val_push(val, step);
            if (val < A.best) A.best = val, better = 1;
        }
        pthread_mutex_unlock(&A.mu);
        if (!isfinite(loss)) {
            pthread_mutex_lock(&A.mu);
            A.running = 0;
            pthread_mutex_unlock(&A.mu);
            note("The loss blew up - lower the learning rate, or make a New brain");
        }
        if (isfinite(val)) {
            pthread_mutex_lock(&A.model_mu);
            if (better) ckpt_save(CKPT ".best", &A.m, &A.s, &A.voc, step, A.best);
            ckpt_save(CKPT, &A.m, &A.s, &A.voc, step, A.best);
            pthread_mutex_unlock(&A.model_mu);
        }
    }
    save_brain();
    gpu_close(g_gpu);
    free(X);
    free(Y);
    return NULL;
}

/* ---------- chat ---------- */

static void *chat_worker(void *arg)
{
    int idx = (int)(intptr_t)arg;
    char conv[2400] = "";
    pthread_mutex_lock(&A.mu);
    for (int i = idx - 8 < 1 ? 1 : idx - 8; i < idx; i++) { /* message 0 is the greeting */
        size_t k = strlen(conv);
        snprintf(conv + k, sizeof conv - k, "%s: %s\n", A.chat[i].ai ? "AI" : "You", A.chat[i].text);
    }
    size_t k = strlen(conv);
    snprintf(conv + k, sizeof conv - k, "AI:");
    double temp = A.creativity;
    pthread_mutex_unlock(&A.mu);

    uint64_t rng = (uint64_t)(now() * 1e6) | 1;
    int buf[3000], n, made = 0, id;
    pthread_mutex_lock(&A.model_mu);
    id = A.brain_id;
    n = A.have_model ? data_encode(&A.voc, conv, buf, 2600) : 0;
    pthread_mutex_unlock(&A.model_mu);
    for (int i = 0; n > 0 && i < 320 && n < 3000; i++) {
        pthread_mutex_lock(&A.model_mu);
        if (!A.have_model || A.brain_id != id) {
            pthread_mutex_unlock(&A.model_mu);
            break;
        }
        int T = A.m.c.T, start = n > T ? n - T : 0;
        buf[n] = model_sample(&A.m, A.gw, buf + start, n - start, temp, &rng);
        char ch = A.voc.itos[buf[n++]];
        pthread_mutex_unlock(&A.model_mu);
        if (ch == '\n') {
            if (made >= 30) break;
            ch = ' ';
        }
        if (made == 0 && ch == ' ') continue;
        pthread_mutex_lock(&A.mu);
        size_t len = strlen(A.chat[idx].text);
        if (len + 1 < MSGLEN) A.chat[idx].text[len] = ch, A.chat[idx].text[len + 1] = 0;
        pthread_mutex_unlock(&A.mu);
        made++;
    }
    pthread_mutex_lock(&A.mu);
    if (!A.chat[idx].text[0]) snprintf(A.chat[idx].text, MSGLEN, "...");
    A.generating = 0;
    pthread_mutex_unlock(&A.mu);
    return NULL;
}

static void add_msg(int ai, const char *text)
{
    if (A.nchat == MAXMSG) {
        memmove(A.chat + 1, A.chat + 2, sizeof(Msg) * (MAXMSG - 2)); /* keep the greeting */
        A.nchat--;
    }
    A.chat[A.nchat].ai = ai;
    snprintf(A.chat[A.nchat].text, MSGLEN, "%s", text);
    A.nchat++;
}

/* ---------- drawing helpers ---------- */

static Font F;

static void txt(const char *s, float x, float y, float sz, Color c) { DrawTextEx(F, s, (Vector2){x, y}, sz, 0, c); }
static float tw(const char *s, float sz) { return MeasureTextEx(F, s, sz, 0).x; }
static int hover(Rectangle r) { return CheckCollisionPointRec(GetMousePosition(), r); }
static int clicked(Rectangle r) { return hover(r) && IsMouseButtonPressed(MOUSE_BUTTON_LEFT); }

static void txt_fit(const char *s, float x, float y, float sz, float maxw, Color c)
{
    char buf[256];
    snprintf(buf, sizeof buf, "%s", s);
    if (tw(buf, sz) <= maxw) return txt(buf, x, y, sz, c);
    for (int n = (int)strlen(buf); n > 1; n--) {
        snprintf(buf + n - 1, sizeof buf - n + 1, "..");
        if (tw(buf, sz) <= maxw) break;
        buf[n - 1] = 0;
    }
    txt(buf, x, y, sz, c);
}

static void panel(Rectangle r, const char *title)
{
    DrawRectangleRounded(r, 0.03f, 8, PANEL);
    DrawRectangleRoundedLinesEx(r, 0.03f, 8, 1, EDGE);
    if (title) txt(title, r.x + 14, r.y + 10, 18, ACC);
}

enum { ICON_NONE, ICON_PLAY, ICON_PAUSE, ICON_STOP };

static void icon(int kind, float x, float y, float s, Color c)
{
    if (kind == ICON_PLAY) DrawTriangle((Vector2){x, y - s}, (Vector2){x, y + s}, (Vector2){x + 1.7f * s, y}, c);
    if (kind == ICON_PAUSE) DrawRectangle(x, y - s, s * 0.6f, 2 * s, c), DrawRectangle(x + s, y - s, s * 0.6f, 2 * s, c);
    if (kind == ICON_STOP) DrawRectangle(x, y - s, 2 * s, 2 * s, c);
}

static int button(Rectangle r, const char *label, Color col, int ic, int active)
{
    int h = hover(r);
    Color fill = Fade(col, active ? 0.45f : h ? 0.3f : 0.14f);
    DrawRectangleRounded(r, 0.35f, 8, fill);
    DrawRectangleRoundedLinesEx(r, 0.35f, 8, active ? 2 : 1, Fade(col, active || h ? 1 : 0.6f));
    float sz = r.height > 34 ? 18 : 15, w = tw(label, sz) + (ic ? 24 : 0), x = r.x + (r.width - w) / 2;
    if (ic) icon(ic, x + 2, r.y + r.height / 2, 6, active || h ? FG : col), x += 24;
    txt(label, x, r.y + (r.height - sz) / 2, sz, active || h ? FG : Fade(FG, 0.85f));
    return h && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

/* A labelled value with - and + buttons. Returns -1, 0 or +1. */
static int stepper(float x, float y, float w, const char *label, const char *value)
{
    txt(label, x, y + 4, 15, DIM);
    Rectangle minus = {x + w - 62, y, 28, 24}, plus = {x + w - 30, y, 28, 24};
    txt(value, minus.x - 8 - tw(value, 15), y + 4, 15, FG);
    int r = 0;
    if (button(minus, "-", ACC, 0, 0)) r = -1;
    if (button(plus, "+", ACC, 0, 0)) r = +1;
    return r;
}

static int chooser(float x, float y, float w, const char *label, const char *value)
{
    txt(label, x, y + 4, 15, DIM);
    char v[64];
    snprintf(v, sizeof v, "%s  >", value);
    return button((Rectangle){x + w - 120, y, 118, 24}, v, ACC, 0, 0);
}

static Color phase_col(float ph, float bright)
{
    float h = (ph + PI) / (2 * PI) * 360;
    return ColorFromHSV(h, 0.6f, fminf(1, fmaxf(0.15f, bright)));
}

static const char *show_char(unsigned char c)
{
    static char buf[8];
    if (c == ' ') return "_";
    if (c == '\n') return "\\n";
    if (c == '\t') return "\\t";
    if (c < 32 || c > 126) return "?";
    buf[0] = c, buf[1] = 0;
    return buf;
}

/* ---------- the brain picture ---------- */

static void draw_brain(Rectangle r, const Inspect *in, const char *probe)
{
    panel(r, "Brain - live view of what it computes");
    if (in->n == 0) {
        const char *m = "Load a text and the brain appears here";
        txt(m, r.x + (r.width - tw(m, 18)) / 2, r.y + r.height / 2, 18, DIM);
        return;
    }
    int n = in->n, L = in->L;
    float top = r.y + 64, bot = r.y + r.height - 34, rowh = (bot - top) / n;
    float x = r.x + 16, xc = x, xw = x + 28, outw = 168, xo = r.x + r.width - outw - 12;
    float xb0 = xw + 80, bw = (xo - 18 - xb0) / L;
    float hy = r.y + 40;
    txt("text", xc, hy, 13, DIM);
    txt("waves", xw + 8, hy, 13, DIM);
    for (int l = 0; l < L; l++) {
        char h[48];
        snprintf(h, sizeof h, "block %d: attention | neurons", l + 1);
        txt_fit(h, xb0 + l * bw + 6, hy, 13, bw - 8, DIM);
    }
    txt("next character (Born rule)", xo, hy, 13, DIM);

    float amax = 1e-9f;
    for (int t = 0; t < n; t++)
        for (int k = 0; k < 4; k++) amax = fmaxf(amax, in->emb_amp[t][k]);
    float rad = fminf(rowh * 0.42f, 8);
    for (int t = 0; t < n; t++) {
        float y = top + rowh * (t + 0.5f);
        txt(show_char(in->ch[t]), xc, y - 8, 15, t == n - 1 ? ACC : FG);
        for (int k = 0; k < 4; k++) {
            float cx = xw + 8 + k * (2 * rad + 4), a = in->emb_amp[t][k] / amax * rad, ph = in->emb_ph[t][k];
            DrawCircleLines(cx, y, rad, Fade(EDGE, 0.9f));
            DrawLineEx((Vector2){cx, y}, (Vector2){cx + a * cosf(ph), y - a * sinf(ph)}, 2, phase_col(ph, 1));
        }
    }

    float nmax = 1e-9f;
    for (int l = 0; l < L; l++)
        for (int k = 0; k < INS_N; k++) nmax = fmaxf(nmax, in->act_amp[l][k]);
    for (int l = 0; l < L; l++) {
        float xa = xb0 + l * bw + 6, xb = xa + bw * 0.52f, xn = xa + bw * 0.78f;
        Color tint = ColorFromHSV(200 + 40 * l, 0.55f, 1);
        for (int t = 0; t < n; t++)
            for (int s = 0; s <= t; s++) {
                float w = in->att[l][t][s];
                if (w < 0.03f) continue;
                DrawLineEx((Vector2){xa, top + rowh * (s + 0.5f)}, (Vector2){xb, top + rowh * (t + 0.5f)}, 0.6f + 3 * w,
                           Fade(tint, 0.15f + 0.85f * w));
            }
        for (int t = 0; t < n; t++) {
            float y = top + rowh * (t + 0.5f);
            DrawCircle(xa, y, 2.5f, DIM);
            DrawCircle(xb, y, 2.5f, t == n - 1 ? ACC : DIM);
        }
        float nh = (bot - top) / INS_N, nr = fminf(nh * 0.36f, 9), yl = top + rowh * (n - 0.5f);
        for (int k = 0; k < INS_N; k++) {
            float y = top + nh * (k + 0.5f), a = in->act_amp[l][k] / nmax;
            DrawLineEx((Vector2){xb, yl}, (Vector2){xn - nr, y}, 1, Fade(ACC, 0.06f + 0.25f * a));
            if (k % 2 == 0 && k + 1 < INS_N) { /* a unitary gate mixes this neuron pair */
                float g = fabsf(sinf(in->gate[l][k / 2]));
                DrawLineEx((Vector2){xn + nr + 4, y}, (Vector2){xn + nr + 4, y + nh}, 3, Fade(WARN, 0.15f + 0.85f * g));
            }
            DrawCircle(xn, y, nr, phase_col(in->act_ph[l][k], 0.25f + 0.75f * a));
            DrawCircleLines(xn, y, nr, Fade(FG, 0.25f));
        }
    }

    for (int i = 0; i < 8; i++) {
        float y = top + i * fminf(30, (bot - top) / 8), p = in->top_p[i];
        char lab[16], pct[16];
        snprintf(lab, sizeof lab, "'%s'", show_char((unsigned char)in->top_ch[i]));
        snprintf(pct, sizeof pct, "%.0f%%", p * 100);
        txt(lab, xo, y + 2, 15, i == 0 ? FG : DIM);
        float bwid = (outw - 86) * p / fmaxf(in->top_p[0], 1e-6f);
        DrawRectangleRounded((Rectangle){xo + 40, y + 2, fmaxf(bwid, 2), 18}, 0.3f, 4, i == 0 ? ACC : Fade(ACC, 0.35f));
        txt(pct, xo + 46 + bwid, y + 3, 14, i == 0 ? FG : DIM);
    }
    char cap[200];
    snprintf(cap, sizeof cap, "reading \"%s\"   colours = phase of each complex number, brightness = amplitude, "
                              "yellow bars = gate strength",
             probe);
    txt_fit(cap, r.x + 16, bot + 10, 13, r.width - 32, DIM);
}

/* ---------- loss chart ---------- */

static void draw_chart(Rectangle r, const float *hl, const long *hs, int nh, const float *vl, const long *vs, int nv)
{
    panel(r, "Learning - loss goes down as it learns");
    Rectangle a = {r.x + 52, r.y + 40, r.width - 70, r.height - 70};
    if (nh < 2) {
        txt("Press START and watch the loss fall", a.x + a.width / 2 - 130, a.y + a.height / 2 - 8, 16, DIM);
        return;
    }
    float lo = 1e9f, hi = -1e9f;
    for (int i = 0; i < nh; i++) lo = fminf(lo, hl[i]), hi = fmaxf(hi, hl[i]);
    for (int i = 0; i < nv; i++) lo = fminf(lo, vl[i]), hi = fmaxf(hi, vl[i]);
    float pad = (hi - lo) * 0.08f + 0.01f;
    lo -= pad, hi += pad;
    long s0 = hs[0], s1 = hs[nh - 1] > s0 ? hs[nh - 1] : s0 + 1;
#define PX(s) (a.x + (float)((s) - s0) / (s1 - s0) * a.width)
#define PY(v) (a.y + a.height - ((v) - lo) / (hi - lo) * a.height)
    for (int i = 0; i <= 3; i++) {
        float v = lo + (hi - lo) * i / 3, y = PY(v);
        char b[16];
        snprintf(b, sizeof b, "%.2f", v);
        DrawLineEx((Vector2){a.x, y}, (Vector2){a.x + a.width, y}, 1, Fade(EDGE, 0.7f));
        txt(b, r.x + 12, y - 7, 13, DIM);
    }
    float sm = hl[0], prev = sm, k = fminf(0.5f, 20.0f / nh);
    for (int i = 1; i < nh; i++) {
        DrawLineEx((Vector2){PX(hs[i - 1]), PY(hl[i - 1])}, (Vector2){PX(hs[i]), PY(hl[i])}, 1, Fade(ACC, 0.25f));
        sm += k * (hl[i] - sm);
        DrawLineEx((Vector2){PX(hs[i - 1]), PY(prev)}, (Vector2){PX(hs[i]), PY(sm)}, 2.5f, ACC);
        prev = sm;
    }
    for (int i = 0; i < nv; i++) {
        if (vs[i] < s0) continue;
        if (i > 0 && vs[i - 1] >= s0)
            DrawLineEx((Vector2){PX(vs[i - 1]), PY(vl[i - 1])}, (Vector2){PX(vs[i]), PY(vl[i])}, 2, WARN);
        DrawCircle(PX(vs[i]), PY(vl[i]), 3, WARN);
    }
    char b[48];
    snprintf(b, sizeof b, "step %ld", s0);
    txt(b, a.x, a.y + a.height + 8, 13, DIM);
    snprintf(b, sizeof b, "step %ld", s1);
    txt(b, a.x + a.width - tw(b, 13), a.y + a.height + 8, 13, DIM);
    DrawRectangle(a.x + a.width - 210, a.y - 26, 14, 4, ACC);
    txt("training", a.x + a.width - 192, a.y - 33, 13, DIM);
    DrawRectangle(a.x + a.width - 120, a.y - 26, 14, 4, WARN);
    txt("unseen text", a.x + a.width - 102, a.y - 33, 13, DIM);
#undef PX
#undef PY
}

/* ---------- chat panel ---------- */

static int wrap(const char *s, float width, float sz, char lines[][200], int maxl)
{
    int nl = 0;
    char cur[200] = "";
    while (*s && nl < maxl) {
        const char *e = s;
        while (*e && *e != ' ') e++;
        char word[200];
        int wl = (int)(e - s) < 190 ? (int)(e - s) : 190;
        memcpy(word, s, wl);
        word[wl] = 0;
        char test[400];
        snprintf(test, sizeof test, "%s%s%s", cur, cur[0] ? " " : "", word);
        if (tw(test, sz) <= width || !cur[0]) {
            snprintf(cur, sizeof cur, "%s", test);
        } else {
            snprintf(lines[nl++], 200, "%s", cur);
            snprintf(cur, sizeof cur, "%s", word);
        }
        s = *e ? e + 1 : e;
    }
    if (cur[0] && nl < maxl) snprintf(lines[nl++], 200, "%s", cur);
    return nl;
}

static float chat_scroll;
static char input[240];

static void draw_chat(Rectangle r, const Msg *msgs, int nmsg, int generating, double creativity, int *send)
{
    panel(r, "Chat with the brain");
    Rectangle box = {r.x + 10, r.y + 40, r.width - 20, r.height - 128};
    float sz = 16, lh = 20, inner = box.width - 40;
    static char lines[48][200];
    float total = 0;
    for (int i = 0; i < nmsg; i++) total += wrap(msgs[i].text, inner, sz, lines, 48) * lh + 34;
    if (hover(box)) chat_scroll += GetMouseWheelMove() * 40;
    float maxs = fmaxf(0, total - box.height);
    chat_scroll = fminf(fmaxf(chat_scroll, 0), maxs);
    BeginScissorMode(box.x, box.y, box.width, box.height);
    float y = box.y + box.height - total + chat_scroll;
    if (total < box.height) y = box.y;
    for (int i = 0; i < nmsg; i++) {
        int nl = wrap(msgs[i].text, inner, sz, lines, 48);
        float h = nl * lh + 12, w = 0;
        for (int j = 0; j < nl; j++) w = fmaxf(w, tw(lines[j], sz));
        w += 20;
        float bx = msgs[i].ai ? box.x + 4 : box.x + box.width - w - 4;
        txt(msgs[i].ai ? "AI" : "You", msgs[i].ai ? bx + 2 : bx + w - tw("You", 13) - 2, y, 13, DIM);
        Rectangle b = {bx, y + 16, w, h};
        DrawRectangleRounded(b, 0.25f, 6, msgs[i].ai ? Fade(ACC, 0.12f) : Fade(GOOD, 0.16f));
        for (int j = 0; j < nl; j++) txt(lines[j], bx + 10, y + 22 + j * lh, sz, FG);
        y += h + 22;
    }
    EndScissorMode();

    char c[64];
    snprintf(c, sizeof c, "%.1f", creativity);
    float cy = r.y + r.height - 80;
    int d = stepper(r.x + 14, cy, r.width - 28, "Creativity (sampling temperature)", c);
    if (d) {
        pthread_mutex_lock(&A.mu);
        A.creativity = fminf(1.5f, fmaxf(0.1f, A.creativity + 0.1 * d));
        pthread_mutex_unlock(&A.mu);
    }
    Rectangle in = {r.x + 10, r.y + r.height - 46, r.width - 96, 36};
    DrawRectangleRounded(in, 0.3f, 6, BG);
    DrawRectangleRoundedLinesEx(in, 0.3f, 6, 1, EDGE);
    BeginScissorMode(in.x + 4, in.y, in.width - 8, in.height);
    float iw = tw(input, 16), ix = in.x + 10 - fmaxf(0, iw - (in.width - 26));
    if (input[0]) txt(input, ix, in.y + 9, 16, FG);
    else txt(generating ? "thinking..." : "Type a message, press Enter", in.x + 10, in.y + 9, 16, DIM);
    if (fmod(GetTime(), 1.0) < 0.5) DrawRectangle(ix + iw + 2, in.y + 8, 2, 20, ACC);
    EndScissorMode();
    *send = button((Rectangle){r.x + r.width - 80, r.y + r.height - 46, 70, 36}, "Send", GOOD, 0, 0) ||
            IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_KP_ENTER);
}

/* ---------- hardware panel ---------- */

static void draw_hardware(Rectangle r, const SensorSnap *hw)
{
    panel(r, "Hardware");
    float y = r.y + 40, x = r.x + 14, w = r.width - 28;
    if (!hw->ready) {
        txt("reading sensors...", x, y, 15, DIM);
        return;
    }
    if (hw->cpu >= 0) {
        char v[32];
        snprintf(v, sizeof v, "%.0f%%", hw->cpu);
        txt("CPU load", x, y, 15, DIM);
        txt(v, x + w - tw(v, 15), y, 15, FG);
        DrawRectangleRounded((Rectangle){x, y + 20, w, 5}, 1, 4, Fade(EDGE, 0.8f));
        DrawRectangleRounded((Rectangle){x, y + 20, w * hw->cpu / 100, 5}, 1, 4, ACC);
        y += 32;
    }
    int shown = 0;
    for (int i = 0; i < hw->n && y < r.y + r.height - 26; i++) {
        const Sensor *s = &hw->s[i];
        char v[32];
        float frac;
        Color col = ACC;
        if (s->kind == SENSOR_TEMP) {
            snprintf(v, sizeof v, "%.0f \xc2\xb0""C", s->value);
            frac = (s->value - 25) / 75;
            col = s->value >= 85 ? BAD : s->value >= 70 ? WARN : GOOD;
        } else if (s->kind == SENSOR_FAN) {
            snprintf(v, sizeof v, "%.0f RPM", s->value);
            frac = s->value / 5000;
        } else {
            snprintf(v, sizeof v, "%.0f%%", s->value);
            frac = s->value / 100;
        }
        frac = fminf(1, fmaxf(0.02f, frac));
        txt_fit(s->label, x, y, 15, w - 80, DIM);
        txt(v, x + w - tw(v, 15), y, 15, FG);
        DrawRectangleRounded((Rectangle){x, y + 20, w, 5}, 1, 4, Fade(EDGE, 0.8f));
        DrawRectangleRounded((Rectangle){x, y + 20, w * frac, 5}, 1, 4, col);
        y += 32;
        shown++;
    }
    if (!shown) txt_fit("No temperature or fan sensors found", x, y, 14, w, DIM);
    else if (y < r.y + r.height - 26) {
        int fans = 0;
        for (int i = 0; i < hw->n; i++) fans += hw->s[i].kind == SENSOR_FAN;
        if (!fans) txt_fit("No fan sensor exposed (try: sudo sensors-detect)", x, y, 13, w, DIM);
    }
}

/* ---------- main ---------- */

static void load_font(void)
{
    const char *paths[] = {"/usr/share/fonts/truetype/fira/FiraSans-Regular.ttf",
                           "/usr/share/fonts/opentype/fira/FiraSans-Regular.otf",
                           "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                           "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                           "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
                           "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"};
    int cps[224];
    for (int i = 0; i < 224; i++) cps[i] = 32 + i;
    for (size_t i = 0; i < sizeof paths / sizeof *paths; i++)
        if (FileExists(paths[i])) {
            F = LoadFontEx(paths[i], 40, cps, 224);
            if (F.glyphCount > 0) {
                SetTextureFilter(F.texture, TEXTURE_FILTER_BILINEAR);
                return;
            }
        }
    F = GetFontDefault();
}

int main(void)
{
    pthread_mutex_init(&A.mu, NULL);
    pthread_mutex_init(&A.model_mu, NULL);
    A.new_cfg = (Config){.D = 64, .H = 4, .L = 2, .T = 64, .R = 2};
    A.o = (OptCfg){.type = OPT_ADAMW, .sched = SCHED_COSINE, .lr = 3e-3, .min_lr = 3e-4, .beta1 = 0.9,
                   .beta2 = 0.99, .wd = 0.01, .clip = 1.0, .warmup = 100, .steps = 5000};
    A.batch = 16, A.eval_every = 250, A.creativity = 0.8, A.best = INFINITY, A.loss = A.val = NAN, A.hstride = 1;
    snprintf(A.probe, sizeof A.probe, "To be, or not to be, that is");
    add_msg(1, "Hi! I'm a tiny quantum brain. I only know the texts you teach me, so I talk like them. "
               "Press START to teach me more, then ask me something.");
    mkdir(TEXT_DIR, 0755);
    scan_files();

    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1440, 880, "QuantumGPT");
    SetWindowMinSize(1180, 720);
    SetTargetFPS(60);
    SetExitKey(KEY_NULL);
    load_font();

    sensors_start();
    pthread_t tr;
    pthread_create(&tr, NULL, trainer, NULL);
    double last_scan = 0;
    int confirm_reset = 0, confirm_del = -1;
    double confirm_t = 0;
    const char *snap = getenv("QGPT_SNAP");
    int snap_frames = getenv("QGPT_SNAP_FRAMES") ? atoi(getenv("QGPT_SNAP_FRAMES")) : 120, frame = 0;
    if (snap && getenv("QGPT_SNAP_RUN")) A.running = 1;

    while (!WindowShouldClose()) {
        if (GetTime() - last_scan > 1.5) scan_files(), last_scan = GetTime();
        if (IsFileDropped()) {
            FilePathList fl = LoadDroppedFiles();
            for (unsigned i = 0; i < fl.count; i++) import_file(fl.paths[i]);
            UnloadDroppedFiles(fl);
            scan_files();
        }
        for (int ch = GetCharPressed(); ch > 0; ch = GetCharPressed())
            if (ch >= 32 && ch < 127 && strlen(input) + 1 < sizeof input) input[strlen(input) + 1] = 0, input[strlen(input)] = (char)ch;
        if ((IsKeyPressed(KEY_BACKSPACE) || IsKeyPressedRepeat(KEY_BACKSPACE)) && input[0]) input[strlen(input) - 1] = 0;
        if (GetTime() - confirm_t > 3) confirm_reset = 0, confirm_del = -1;
        int test_send = snap && getenv("QGPT_TEST_CHAT") && frame == 30;
        if (test_send) snprintf(input, sizeof input, "%s", getenv("QGPT_TEST_CHAT"));

        /* snapshot of shared state for this frame */
        static float hl[HMAX], vl[HMAX];
        static long hs[HMAX], vs[HMAX];
        static Msg msgs[MAXMSG];
        static Inspect ins;
        pthread_mutex_lock(&A.mu);
        int nh = A.nh, nv = A.nv, running = A.running, nchat = A.nchat, generating = A.generating, nfiles = A.nfiles;
        memcpy(hl, A.hl, sizeof(float) * nh), memcpy(hs, A.hs, sizeof(long) * nh);
        memcpy(vl, A.vl, sizeof(float) * nv), memcpy(vs, A.vs, sizeof(long) * nv);
        memcpy(msgs, A.chat, sizeof(Msg) * nchat);
        ins = A.ins;
        TextFile files[MAXF];
        memcpy(files, A.files, sizeof(TextFile) * nfiles);
        long step = A.step;
        double loss = A.loss, val = A.val, best = A.best, tps = A.tps, creativity = A.creativity;
        OptCfg o = A.o;
        int batch = A.batch;
        Config nc = A.new_cfg, cur = A.cur;
        char note_s[220], sample[400], probe[128];
        snprintf(note_s, sizeof note_s, "%s", now() - A.note_t < 8 ? A.note : "");
        snprintf(sample, sizeof sample, "%s", A.sample);
        snprintf(probe, sizeof probe, "%s", A.probe);
        int have_model = A.have_model, use_gpu = A.use_gpu, gpu_ok = A.gpu_ok;
        char dev_label[200];
        snprintf(dev_label, sizeof dev_label, "%s", A.dev_label);
        pthread_mutex_unlock(&A.mu);
        SensorSnap hw;
        sensors_get(&hw);

        int W = GetScreenWidth(), H = GetScreenHeight();
        BeginDrawing();
        ClearBackground(BG);

        /* top bar */
        txt("QuantumGPT", 18, 12, 28, FG);
        char dev[160];
        if (use_gpu) snprintf(dev, sizeof dev, "training on the GPU: %s", dev_label);
        else if (gpu_ok) snprintf(dev, sizeof dev, "training on CPU, %d threads  |  GPU available: %s", learn_threads(0), dev_label);
        else snprintf(dev, sizeof dev, "training on CPU, %d threads  |  no GPU: %s", learn_threads(0), dev_label);
        txt(dev, 20, 44, 14, DIM);
        float bx = W - 12;
        Rectangle rstop = {bx -= 170, 12, 160, 40}, rpause = {bx -= 132, 12, 122, 40}, rstart = {bx -= 132, 12, 122, 40};
        if (button(rstart, "START", GOOD, ICON_PLAY, running)) {
            pthread_mutex_lock(&A.mu);
            A.running = 1;
            if (A.step >= A.o.steps) A.o.steps = (A.step / 1000 + 5) * 1000;
            pthread_mutex_unlock(&A.mu);
            if (!A.have_data) note("Add a .txt file first - drag it onto the window");
        }
        if (button(rpause, "PAUSE", WARN, ICON_PAUSE, !running && step > 0)) {
            pthread_mutex_lock(&A.mu);
            A.running = 0;
            pthread_mutex_unlock(&A.mu);
        }
        if (button(rstop, "STOP & SAVE", BAD, ICON_STOP, 0)) {
            pthread_mutex_lock(&A.mu);
            A.running = 0, A.want_save = 1;
            pthread_mutex_unlock(&A.mu);
        }
        Color pill = running ? GOOD : WARN;
        const char *ps = running ? "LEARNING" : have_model ? "PAUSED" : "NO BRAIN YET";
        Rectangle pr = {bx - 20 - tw(ps, 15) - 28, 20, tw(ps, 15) + 28, 26};
        DrawRectangleRounded(pr, 1, 8, Fade(pill, 0.18f));
        DrawCircle(pr.x + 12, pr.y + 13, 4, pill);
        txt(ps, pr.x + 20, pr.y + 5, 15, pill);
        if (note_s[0]) txt_fit(note_s, 290, 20, 15, pr.x - 310, FG);

        float top = 68, lw = 310, rw = 380, gap = 12;
        Rectangle left = {12, top, lw, H - top - 12}, right = {W - rw - 12, top, rw, H - top - 12};
        Rectangle center = {left.x + lw + gap, top, W - lw - rw - 24 - 2 * gap, H - top - 12};

        /* texts */
        float th = 54 + 26 * (nfiles < 6 ? (nfiles ? nfiles : 1) : 6) + 44;
        Rectangle rt = {left.x, left.y, lw, th};
        panel(rt, "Training texts");
        float y = rt.y + 40;
        if (!nfiles) txt("Drag .txt files onto this window", rt.x + 14, y + 4, 15, DIM), y += 26;
        for (int i = 0; i < nfiles && i < 6; i++, y += 26) {
            Rectangle cb = {rt.x + 14, y + 3, 16, 16};
            DrawRectangleRounded(cb, 0.3f, 4, files[i].on ? ACC : BG);
            DrawRectangleRoundedLinesEx(cb, 0.3f, 4, 1, ACC);
            char sz[24];
            snprintf(sz, sizeof sz, files[i].size > 1 << 20 ? "%.1f MB" : "%.0f KB",
                     files[i].size > 1 << 20 ? files[i].size / 1048576.0 : files[i].size / 1024.0);
            txt_fit(files[i].name, rt.x + 38, y + 2, 15, lw - 150, files[i].on ? FG : DIM);
            txt(sz, rt.x + lw - 50 - tw(sz, 13), y + 4, 13, DIM);
            Rectangle del = {rt.x + lw - 40, y, 26, 22};
            if (clicked((Rectangle){cb.x, y, lw - 60, 22})) {
                pthread_mutex_lock(&A.mu);
                for (int j = 0; j < A.nfiles; j++)
                    if (!strcmp(A.files[j].name, files[i].name)) A.files[j].on = !A.files[j].on;
                A.want_reload = 1;
                save_disabled();
                pthread_mutex_unlock(&A.mu);
            }
            if (button(del, confirm_del == i ? "?" : "x", BAD, 0, confirm_del == i)) {
                if (confirm_del == i) {
                    char path[300];
                    snprintf(path, sizeof path, TEXT_DIR "/%s", files[i].name);
                    remove(path);
                    note("Deleted %s", files[i].name);
                    confirm_del = -1;
                    scan_files();
                } else
                    confirm_del = i, confirm_t = GetTime(), note("Click ? again to delete %s", files[i].name);
            }
        }
        if (nfiles > 6) txt_fit("(more files in the texts folder)", rt.x + 14, y - 4, 12, lw - 28, DIM);
        if (button((Rectangle){rt.x + 14, rt.y + th - 38, 136, 28}, "Add files...", ACC, 0, 0)) {
            pthread_t t;
            if (pthread_create(&t, NULL, pick_files, NULL) == 0) pthread_detach(t);
        }
        if (button((Rectangle){rt.x + 160, rt.y + th - 38, 136, 28}, "Open folder", ACC, 0, 0))
            if (system("xdg-open " TEXT_DIR " >/dev/null 2>&1 &")) note("Could not open the folder");

        /* learning settings */
        Rectangle rl = {left.x, rt.y + th + gap, lw, 243};
        panel(rl, "Learning");
        float sx = rl.x + 14, sw = lw - 28;
        y = rl.y + 40;
        char v[48];
        int d;
        pthread_mutex_lock(&A.mu);
        if (chooser(sx, y, sw, "Device", use_gpu ? "GPU" : "CPU")) {
            if (A.gpu_ok) A.use_gpu = !A.use_gpu;
            else snprintf(A.note, sizeof A.note, "No GPU: %s", A.dev_label), A.note_t = now();
        }
        if (chooser(sx, y += 29, sw, "Optimiser", opt_name(o.type))) A.o.type = (A.o.type + 1) % 4;
        if (chooser(sx, y += 29, sw, "Schedule", sched_name(o.sched))) A.o.sched = (A.o.sched + 1) % 3;
        snprintf(v, sizeof v, "%.1e", o.lr);
        if ((d = stepper(sx, y += 29, sw, "Learning rate", v))) {
            double k = d > 0 ? 1.5 : 1 / 1.5;
            A.o.lr *= k, A.o.min_lr *= k;
        }
        snprintf(v, sizeof v, "%ld", o.steps);
        if ((d = stepper(sx, y += 29, sw, "Steps", v))) A.o.steps = d > 0 ? A.o.steps + 1000 : (A.o.steps > 1000 ? A.o.steps - 1000 : A.o.steps);
        snprintf(v, sizeof v, "%d", batch);
        if ((d = stepper(sx, y += 29, sw, "Batch", v))) A.batch = d > 0 ? (A.batch < 128 ? A.batch * 2 : 128) : (A.batch > 1 ? A.batch / 2 : 1);
        snprintf(v, sizeof v, "%.1f", o.anneal);
        if ((d = stepper(sx, y += 29, sw, "Annealing noise", v))) A.o.anneal = fmax(0, A.o.anneal + 0.1 * d);
        pthread_mutex_unlock(&A.mu);

        /* new brain */
        Rectangle rn = {left.x, rl.y + rl.height + gap, lw, 196};
        panel(rn, "New brain");
        char shape[96];
        if (have_model)
            snprintf(shape, sizeof shape, "now: width %d, %d blocks, context %d", cur.D, cur.L, cur.T);
        else
            snprintf(shape, sizeof shape, "no brain yet");
        txt_fit(shape, rn.x + 120, rn.y + 13, 13, lw - 134, DIM);
        y = rn.y + 38;
        pthread_mutex_lock(&A.mu);
        snprintf(v, sizeof v, "%d", nc.D);
        if ((d = stepper(sx, y, sw, "Width", v))) A.new_cfg.D = d > 0 ? (nc.D < 256 ? nc.D + 32 : 256) : (nc.D > 32 ? nc.D - 32 : 32);
        snprintf(v, sizeof v, "%d", nc.L);
        if ((d = stepper(sx, y += 28, sw, "Blocks", v))) A.new_cfg.L = d > 0 ? (nc.L < 8 ? nc.L + 1 : 8) : (nc.L > 1 ? nc.L - 1 : 1);
        snprintf(v, sizeof v, "%d", nc.T);
        if ((d = stepper(sx, y += 28, sw, "Context", v))) A.new_cfg.T = d > 0 ? (nc.T < 256 ? nc.T * 2 : 256) : (nc.T > 16 ? nc.T / 2 : 16);
        snprintf(v, sizeof v, "%d", nc.R);
        if ((d = stepper(sx, y += 28, sw, "Circuit depth", v))) A.new_cfg.R = d > 0 ? (nc.R < 6 ? nc.R + 1 : 6) : (nc.R > 0 ? nc.R - 1 : 0);
        A.new_cfg.H = 4;
        pthread_mutex_unlock(&A.mu);
        if (button((Rectangle){sx, y += 32, sw, 30}, confirm_reset ? "Click again: replaces the current brain" : "Create new brain",
                   BAD, 0, confirm_reset)) {
            if (confirm_reset || !have_model) {
                pthread_mutex_lock(&A.mu);
                A.want_reset = 1;
                pthread_mutex_unlock(&A.mu);
                confirm_reset = 0;
            } else
                confirm_reset = 1, confirm_t = GetTime();
        }

        /* hardware */
        Rectangle rh = {left.x, rn.y + rn.height + gap, lw, left.y + left.height - (rn.y + rn.height + gap)};
        if (rh.height > 60) draw_hardware(rh, &hw);

        /* center: brain + chart */
        Rectangle rb = {center.x, center.y, center.width, center.height * 0.6f};
        Rectangle rc = {center.x, rb.y + rb.height + gap, center.width, center.height - rb.height - gap};
        draw_brain(rb, &ins, probe);
        draw_chart(rc, hl, hs, nh, vl, vs, nv);

        /* right: stats + chat */
        Rectangle rs = {right.x, right.y, rw, 150};
        panel(rs, NULL);
        const char *names[4] = {"step", "loss", "unseen text loss", "speed"};
        char vals[4][48];
        snprintf(vals[0], 48, "%ld", step);
        snprintf(vals[1], 48, isfinite(loss) ? "%.3f" : "-", loss);
        if (isfinite(val)) snprintf(vals[2], 48, "%.3f", val);
        else if (isfinite(best)) snprintf(vals[2], 48, "best %.3f", best);
        else snprintf(vals[2], 48, "-");
        snprintf(vals[3], 48, running && tps > 0 ? "%.0f ch/s" : "-", tps);
        for (int i = 0; i < 4; i++) {
            float tx = rs.x + 16 + (i % 2) * (rw / 2), ty = rs.y + 12 + (i / 2) * 50;
            txt(names[i], tx, ty, 13, DIM);
            txt(vals[i], tx, ty + 16, 24, FG);
        }
        txt_fit(sample[0] ? sample : "(a sample of its writing appears here every 250 steps)", rs.x + 16, rs.y + 118,
                13, rw - 32, sample[0] ? ACC : DIM);
        Rectangle rch = {right.x, rs.y + rs.height + gap, rw, right.height - rs.height - gap};
        int send = 0;
        draw_chat(rch, msgs, nchat, generating, creativity, &send);
        if ((send || test_send) && input[0]) {
            if (generating) note("Wait for the brain to finish its answer");
            else if (!have_model) note("There is no brain yet - add a .txt file first");
            else {
                pthread_mutex_lock(&A.mu);
                add_msg(0, input);
                add_msg(1, "");
                A.generating = 1;
                snprintf(A.probe, sizeof A.probe, "%s", input);
                A.want_inspect = 1;
                int idx = A.nchat - 1;
                pthread_mutex_unlock(&A.mu);
                input[0] = 0;
                chat_scroll = 0;
                pthread_t t;
                if (pthread_create(&t, NULL, chat_worker, (void *)(intptr_t)idx) == 0) pthread_detach(t);
                else A.generating = 0;
            }
        }
        EndDrawing();
        if (snap && ++frame == snap_frames) {
            TakeScreenshot(snap);
            break;
        }
    }

    pthread_mutex_lock(&A.mu);
    A.quit = 1, A.running = 0;
    pthread_mutex_unlock(&A.mu);
    pthread_join(tr, NULL);
    CloseWindow();
    return 0;
}
