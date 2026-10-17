/* sensors.c - temperatures, fan speeds and CPU load for the GUI's hardware panel.
 * Reads Linux hwmon (/sys/class/hwmon), thermal zones and /proc/stat, plus
 * nvidia-smi when an NVIDIA card is present. Polls in its own thread. */
#include "sensors.h"
#include <dirent.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static SensorSnap snap;

static int read_long(const char *path, long *out)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int ok = fscanf(f, "%ld", out) == 1;
    fclose(f);
    return ok ? 0 : -1;
}

static void read_line(const char *path, char *out, size_t cap)
{
    out[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(out, (int)cap, f)) out[strcspn(out, "\n")] = 0;
    fclose(f);
}

static void add(SensorSnap *s, int kind, double value, const char *fmt, const char *a, const char *b)
{
    if (s->n >= SENSOR_MAX) return;
    Sensor *x = &s->s[s->n++];
    x->kind = kind;
    x->value = value;
    snprintf(x->label, sizeof x->label, fmt, a, b);
}

/* Friendly chip names for the usual suspects. */
static const char *chip_name(const char *name)
{
    if (!strcmp(name, "k10temp") || !strcmp(name, "zenpower")) return "CPU";
    if (!strcmp(name, "coretemp")) return "CPU";
    if (!strcmp(name, "amdgpu")) return "GPU";
    if (!strcmp(name, "nouveau")) return "GPU";
    if (!strncmp(name, "nvme", 4)) return "SSD";
    if (!strcmp(name, "acpitz")) return "Board";
    return name;
}

static void scan_hwmon(SensorSnap *s)
{
    DIR *dir = opendir("/sys/class/hwmon");
    if (!dir) return;
    struct dirent *e;
    while ((e = readdir(dir))) {
        if (e->d_name[0] == '.') continue;
        char base[300], path[400], name[64], label[64];
        snprintf(base, sizeof base, "/sys/class/hwmon/%s", e->d_name);
        snprintf(path, sizeof path, "%s/name", base);
        read_line(path, name, sizeof name);
        const char *chip = chip_name(name);
        int temps = 0;
        for (int i = 1; i <= 16; i++) {
            long v;
            snprintf(path, sizeof path, "%s/temp%d_input", base, i);
            if (read_long(path, &v) || v <= 0) continue;
            snprintf(path, sizeof path, "%s/temp%d_label", base, i);
            read_line(path, label, sizeof label);
            if (!strcmp(name, "coretemp") && strncmp(label, "Package", 7)) continue; /* skip per-core */
            if (++temps > 3) break;
            add(s, SENSOR_TEMP, v / 1000.0, label[0] ? "%s %s" : "%s%s", chip, label);
        }
        for (int i = 1; i <= 8; i++) {
            long v;
            snprintf(path, sizeof path, "%s/fan%d_input", base, i);
            if (read_long(path, &v)) continue;
            snprintf(path, sizeof path, "%s/fan%d_label", base, i);
            read_line(path, label, sizeof label);
            char idx[16];
            snprintf(idx, sizeof idx, "fan %d", i);
            add(s, SENSOR_FAN, (double)v, "%s %s", chip, label[0] ? label : idx);
        }
    }
    closedir(dir);
}

static void scan_thermal_zones(SensorSnap *s)
{
    for (int i = 0; i < 4; i++) {
        char path[128], type[64];
        long v;
        snprintf(path, sizeof path, "/sys/class/thermal/thermal_zone%d/temp", i);
        if (read_long(path, &v) || v <= 0) continue;
        snprintf(path, sizeof path, "/sys/class/thermal/thermal_zone%d/type", i);
        read_line(path, type, sizeof type);
        add(s, SENSOR_TEMP, v / 1000.0, "%s%s", type[0] ? type : "zone", "");
    }
}

static void scan_nvidia(SensorSnap *s)
{
    static int missing;
    if (missing) return;
    FILE *p = popen("nvidia-smi --query-gpu=name,temperature.gpu,fan.speed,utilization.gpu "
                    "--format=csv,noheader,nounits 2>/dev/null",
                    "r");
    if (!p) {
        missing = 1;
        return;
    }
    char line[256];
    if (fgets(line, sizeof line, p)) {
        char *name = strtok(line, ","), *temp = strtok(NULL, ","), *fan = strtok(NULL, ","), *use = strtok(NULL, ",");
        if (name && temp) {
            snprintf(s->gpu_name, sizeof s->gpu_name, "%s", name);
            add(s, SENSOR_TEMP, atof(temp), "%s%s", "GPU", "");
            if (fan && atof(fan) >= 0 && strstr(fan, "N/A") == NULL) add(s, SENSOR_PERCENT, atof(fan), "%s%s", "GPU fan", "");
            if (use) add(s, SENSOR_PERCENT, atof(use), "%s%s", "GPU load", "");
        }
    } else
        missing = 1;
    if (pclose(p) != 0) missing = 1;
}

static double cpu_usage(void)
{
    static unsigned long long last_idle, last_total;
    unsigned long long v[10] = {0};
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return -1;
    int got = fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4],
                     &v[5], &v[6], &v[7], &v[8], &v[9]);
    fclose(f);
    if (got < 4) return -1;
    unsigned long long idle = v[3] + v[4], total = 0;
    for (int i = 0; i < 8; i++) total += v[i];
    double use = -1;
    if (last_total && total > last_total) use = 100.0 * (1.0 - (double)(idle - last_idle) / (total - last_total));
    last_idle = idle, last_total = total;
    return use;
}

static void *poll_loop(void *arg)
{
    (void)arg;
    for (;;) {
        SensorSnap s = {0};
        scan_hwmon(&s);
        int temps = 0;
        for (int i = 0; i < s.n; i++) temps += s.s[i].kind == SENSOR_TEMP;
        if (!temps) scan_thermal_zones(&s);
        scan_nvidia(&s);
        s.cpu = cpu_usage();
        s.ready = 1;
        pthread_mutex_lock(&mu);
        snap = s;
        pthread_mutex_unlock(&mu);
        sleep(2);
    }
    return NULL;
}

void sensors_start(void)
{
    pthread_t t;
    if (pthread_create(&t, NULL, poll_loop, NULL) == 0) pthread_detach(t);
}

void sensors_get(SensorSnap *out)
{
    pthread_mutex_lock(&mu);
    *out = snap;
    pthread_mutex_unlock(&mu);
}
