#ifndef UI_H
#define UI_H

#include "raylib.h"
#include "world.h"

extern Font fontTitle, fontUI, fontMono;

typedef enum {
    ACT_NONE, ACT_START, ACT_HOWTO, ACT_SETTINGS, ACT_QUIT, ACT_BACK,
    ACT_RESUME, ACT_ABANDON, ACT_RETRY, ACT_MENU
} UiAction;

void UiInit(void);
void UiUnload(void);

Vector2 UiMeasure(Font f, const char *s, float size);
void UiText(Font f, const char *s, float x, float y, float size, Color c);
void UiTextCentered(Font f, const char *s, float cx, float y, float size, Color c);
void UiPanel(Rectangle r, Color fill, Color edge);
void UiBar(Rectangle r, float frac, Color fill, Color back);
void UiDrawBuddy(Vector2 c, float scale, float talk, float t);
void UiDrawGlooper2D(Vector2 c, float r, Vector2 look, float t, Color col);

UiAction UiMainMenu(float t);
UiAction UiSettingsMenu(float t, bool inGame);
UiAction UiHowTo(float t);
UiAction UiPauseMenu(float t);
UiAction UiEndScreen(float t, const RunStats *s, bool won, bool newBest);

#endif
