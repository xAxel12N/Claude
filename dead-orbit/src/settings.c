// Settings and best scores, stored as key=value text in the XDG data directory.
#include "settings.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

Settings settings;

static void ConfigPath(char *out, size_t n, bool makeDir)
{
    const char *xdg = getenv("XDG_DATA_HOME");
    const char *home = getenv("HOME");
    char dir[512];
    if (xdg && xdg[0]) snprintf(dir, sizeof dir, "%s/dead-orbit", xdg);
    else if (home && home[0]) snprintf(dir, sizeof dir, "%s/.local/share/dead-orbit", home);
    else snprintf(dir, sizeof dir, ".");
    if (makeDir) {
        // create each path component; mkdir on an existing directory is harmless
        char tmp[512];
        snprintf(tmp, sizeof tmp, "%s", dir);
        for (char *p = tmp + 1; *p; p++) {
            if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
        }
        mkdir(tmp, 0755);
    }
    snprintf(out, n, "%s/settings.cfg", dir);
}

void SettingsDefaults(void)
{
    settings = (Settings){
        .mouseSens = 1.0f, .invertY = false, .masterVol = 0.8f, .musicVol = 0.7f,
        .fullscreen = false, .bloom = true, .screenShake = true, .difficulty = DIFF_NORMAL,
    };
}

static float Clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

void SettingsLoad(void)
{
    SettingsDefaults();
    char path[600];
    ConfigPath(path, sizeof path, false);
    FILE *f = fopen(path, "r");
    if (!f) return;
    char key[64];
    float val;
    while (fscanf(f, " %63[^=]=%f", key, &val) == 2) {
        if (!strcmp(key, "mouse_sens")) settings.mouseSens = Clampf(val, 0.2f, 3.0f);
        else if (!strcmp(key, "invert_y")) settings.invertY = val != 0;
        else if (!strcmp(key, "master_volume")) settings.masterVol = Clampf(val, 0, 1);
        else if (!strcmp(key, "music_volume")) settings.musicVol = Clampf(val, 0, 1);
        else if (!strcmp(key, "fullscreen")) settings.fullscreen = val != 0;
        else if (!strcmp(key, "bloom")) settings.bloom = val != 0;
        else if (!strcmp(key, "screen_shake")) settings.screenShake = val != 0;
        else if (!strcmp(key, "difficulty")) settings.difficulty = (int)Clampf(val, 0, DIFF_COUNT - 1);
        else if (!strcmp(key, "best_easy")) settings.best[DIFF_EASY] = (int)val;
        else if (!strcmp(key, "best_normal")) settings.best[DIFF_NORMAL] = (int)val;
        else if (!strcmp(key, "best_nightmare")) settings.best[DIFF_NIGHTMARE] = (int)val;
    }
    fclose(f);
}

void SettingsSave(void)
{
    char path[600];
    ConfigPath(path, sizeof path, true);
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "mouse_sens=%.2f\ninvert_y=%d\nmaster_volume=%.2f\nmusic_volume=%.2f\n",
            settings.mouseSens, settings.invertY, settings.masterVol, settings.musicVol);
    fprintf(f, "fullscreen=%d\nbloom=%d\nscreen_shake=%d\ndifficulty=%d\n",
            settings.fullscreen, settings.bloom, settings.screenShake, settings.difficulty);
    fprintf(f, "best_easy=%d\nbest_normal=%d\nbest_nightmare=%d\n",
            settings.best[DIFF_EASY], settings.best[DIFF_NORMAL], settings.best[DIFF_NIGHTMARE]);
    fclose(f);
}
