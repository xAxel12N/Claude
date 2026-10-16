#ifndef SETTINGS_H
#define SETTINGS_H

#include <stdbool.h>

typedef enum { DIFF_EASY, DIFF_NORMAL, DIFF_NIGHTMARE, DIFF_COUNT } Difficulty;

typedef struct {
    float mouseSens;   // 0.2 .. 3.0
    bool invertY;
    float masterVol;   // 0 .. 1
    float musicVol;    // 0 .. 1
    bool fullscreen;
    bool bloom;
    bool screenShake;
    int difficulty;
    int best[DIFF_COUNT];
} Settings;

extern Settings settings;

void SettingsDefaults(void);
void SettingsLoad(void);
void SettingsSave(void);

#endif
