#ifndef WORLD_H
#define WORLD_H

#include "raylib.h"
#include "settings.h"
#include <stdbool.h>

typedef struct {
    Vector2 look;           // mouse delta in degrees
    float fwd, right, up;   // thruster axes, -1..1
    bool boost, boostPressed, brake;
    bool cut, tether, release, reel;
} Input;

typedef enum { RUN_PLAYING, RUN_DEAD, RUN_WON } RunState;

typedef struct {
    int credits, quota;
    float o2, reserve;
    int deposited, gloopsPopped, mimicsSlain;
    float time;
    int score;
    const char *deathCause;
} RunStats;

void WorldNew(Difficulty d);
RunState WorldUpdate(float dt, const Input *in);
void WorldIdle(float dt);               // animate creatures for the menu backdrop
Camera3D WorldCamera(void);
Camera3D WorldMenuCamera(float t);
void WorldDraw(Camera3D cam, bool drawPlayer);
void WorldDrawHUD(Camera3D cam);
const RunStats *WorldStats(void);
float WorldDanger(void);
float WorldHurt(void);
float WorldFade(void);
void WorldDebugStage(int stage);        // used by --autotest to frame screenshots

#endif
