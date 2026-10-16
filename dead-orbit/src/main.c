// Dead Orbit: zero-g salvage horror-comedy. Entry point, state machine and input.
#include "raylib.h"
#include "rlgl.h"
#include "gfx.h"
#include "world.h"
#include "audio.h"
#include "ui.h"
#include "settings.h"
#include <stdio.h>
#include <string.h>

typedef enum { ST_MENU, ST_HOWTO, ST_SETTINGS, ST_PLAY, ST_PAUSE, ST_END } State;

static State state = ST_MENU, returnTo = ST_MENU;
static bool runWon, newBest, cursorLocked, skipMouse;
static bool fullscreenApplied;

static void LockCursor(bool lock)
{
    if (lock == cursorLocked) return;
    cursorLocked = lock;
    if (lock) { DisableCursor(); skipMouse = true; }
    else EnableCursor();
}

static void ApplyFullscreen(void)
{
    if (settings.fullscreen == fullscreenApplied) return;
    ToggleBorderlessWindowed();
    fullscreenApplied = settings.fullscreen;
}

static Input ReadInput(void)
{
    Input in = { 0 };
    Vector2 md = GetMouseDelta();
    if (skipMouse) { md = (Vector2){ 0, 0 }; skipMouse = false; }
    in.look = md;
    in.fwd = (IsKeyDown(KEY_W) ? 1.0f : 0.0f) - (IsKeyDown(KEY_S) ? 1.0f : 0.0f);
    in.right = (IsKeyDown(KEY_D) ? 1.0f : 0.0f) - (IsKeyDown(KEY_A) ? 1.0f : 0.0f);
    in.up = (IsKeyDown(KEY_SPACE) ? 1.0f : 0.0f) - ((IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_C)) ? 1.0f : 0.0f);
    in.boost = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
    in.boostPressed = IsKeyPressed(KEY_LEFT_SHIFT) || IsKeyPressed(KEY_RIGHT_SHIFT);
    in.brake = IsKeyDown(KEY_X);
    in.cut = IsMouseButtonDown(MOUSE_BUTTON_LEFT);
    in.tether = IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsKeyPressed(KEY_F);
    in.release = IsKeyPressed(KEY_E);
    in.reel = IsKeyDown(KEY_R) || GetMouseWheelMove() > 0;
    return in;
}

static void MakeIcon(void)
{
    Image img = GenImageColor(64, 64, BLANK);
    ImageDrawCircle(&img, 32, 34, 26, (Color){ 120, 235, 90, 255 });
    ImageDrawCircle(&img, 22, 28, 10, WHITE);
    ImageDrawCircle(&img, 43, 26, 7, WHITE);
    ImageDrawCircle(&img, 24, 30, 4, BLACK);
    ImageDrawCircle(&img, 44, 28, 3, BLACK);
    ImageDrawRectangle(&img, 20, 44, 24, 5, (Color){ 45, 6, 18, 255 });
    SetWindowIcon(img);
    UnloadImage(img);
}

static void StartRun(void)
{
    WorldNew((Difficulty)settings.difficulty);
    state = ST_PLAY;
}

static void FinishRun(RunState rs)
{
    runWon = rs == RUN_WON;
    newBest = false;
    if (runWon && WorldStats()->score > settings.best[settings.difficulty]) {
        settings.best[settings.difficulty] = WorldStats()->score;
        newBest = true;
        SettingsSave();
    }
    state = ST_END;
}

static void Capture(const char *dir, const char *name)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    rlDrawRenderBatchActive();
    Image img = LoadImageFromScreen();
    ExportImage(img, path);
    UnloadImage(img);
}

int main(int argc, char **argv)
{
    const char *autotest = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--autotest") && i + 1 < argc) autotest = argv[++i];
        else if (!strcmp(argv[i], "--help")) {
            printf("dead-orbit [--autotest DIR]\n  --autotest DIR  play a scripted demo, save screenshots to DIR, then quit\n");
            return 0;
        }
    }

    SettingsLoad();
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT);
    SetTraceLogLevel(autotest ? LOG_WARNING : LOG_ERROR);
    InitWindow(1600, 900, "Dead Orbit");
    if (!autotest) {
        int m = GetCurrentMonitor();
        int mw = GetMonitorWidth(m), mh = GetMonitorHeight(m);
        if (mw > 0 && mh > 0 && (mw < 1700 || mh < 1000)) SetWindowSize(mw*85/100, mh*85/100);
    } else {
        SetWindowSize(1280, 720);
    }
    SetExitKey(KEY_NULL);
    SetTargetFPS(144);
    MakeIcon();
    AudioInit();
    AudioSetVolumes(settings.masterVol, settings.musicVol);
    GfxInit();
    UiInit();
    WorldNew((Difficulty)settings.difficulty);
    if (!autotest) ApplyFullscreen();

    float menuT = 0;
    int frame = 0;
    bool quit = false;
    while (!quit && !WindowShouldClose()) {
        float dt = autotest ? 1.0f/60 : GetFrameTime();
        if (dt > 0.1f) dt = 0.1f;
        menuT += dt;
        frame++;
        if (!autotest) ApplyFullscreen();
        LockCursor(state == ST_PLAY && !autotest);

        Input scripted = { 0 };
        if (autotest) {
            if (frame == 100) state = ST_HOWTO;
            if (frame == 110) state = ST_SETTINGS;
            if (frame == 120) StartRun();
            if (frame > 120 && frame < 200) { scripted.fwd = 1; scripted.look.x = -2; }
            if (frame == 210) WorldDebugStage(1);
            if (frame > 250 && frame < 275) scripted.cut = true;
            if (frame == 280) state = ST_PAUSE;
            if (frame == 290) { state = ST_PLAY; WorldDebugStage(2); }
        }

        if (state == ST_PLAY) {
            Input in = autotest ? scripted : ReadInput();
            RunState rs = WorldUpdate(dt, &in);
            if (rs != RUN_PLAYING) FinishRun(rs);
            else if (IsKeyPressed(KEY_ESCAPE) || (IsWindowFocused() == false && !autotest)) state = ST_PAUSE;
        } else if (state == ST_MENU || ((state == ST_HOWTO || state == ST_SETTINGS) && returnTo == ST_MENU)) {
            WorldIdle(dt);
        }

        bool inRun = state == ST_PLAY || state == ST_PAUSE || state == ST_END || returnTo == ST_PAUSE;
        if (state == ST_MENU) returnTo = ST_MENU, inRun = false;
        Camera3D cam = inRun ? WorldCamera() : WorldMenuCamera(menuT);
        WorldDraw(cam, inRun);

        BeginDrawing();
        ClearBackground(BLACK);
        float danger = state == ST_PLAY ? WorldDanger() : 0;
        GfxPresent(danger, state == ST_PLAY ? WorldHurt() : 0, inRun ? WorldFade()*0.85f : 0, settings.bloom);
        if (state == ST_PLAY) WorldDrawHUD(cam);

        UiAction act = ACT_NONE;
        switch (state) {
        case ST_MENU: act = UiMainMenu(menuT); break;
        case ST_HOWTO: act = UiHowTo(menuT); break;
        case ST_SETTINGS: act = UiSettingsMenu(menuT, returnTo == ST_PAUSE); break;
        case ST_PAUSE: act = UiPauseMenu(menuT); break;
        case ST_END: act = UiEndScreen(menuT, WorldStats(), runWon, newBest); break;
        case ST_PLAY: break;
        }

        if (autotest) {
            if (frame == 99) Capture(autotest, "01_menu.png");
            if (frame == 109) Capture(autotest, "02_howto.png");
            if (frame == 119) Capture(autotest, "03_settings.png");
            if (frame == 199) Capture(autotest, "04_play.png");
            if (frame == 274) Capture(autotest, "05_creatures.png");
            if (frame == 289) Capture(autotest, "06_pause.png");
            if (state == ST_END) { static int endFrames; if (++endFrames == 30) { Capture(autotest, "07_end.png"); quit = true; } }
            if (frame > 1200) quit = true;
        }
        EndDrawing();

        switch (act) {
        case ACT_START: case ACT_RETRY: StartRun(); break;
        case ACT_HOWTO: returnTo = state == ST_PAUSE ? ST_PAUSE : ST_MENU; state = ST_HOWTO; break;
        case ACT_SETTINGS: returnTo = state == ST_PAUSE ? ST_PAUSE : ST_MENU; state = ST_SETTINGS; break;
        case ACT_BACK: state = returnTo; break;
        case ACT_RESUME: state = ST_PLAY; break;
        case ACT_ABANDON: case ACT_MENU:
            WorldNew((Difficulty)settings.difficulty);
            returnTo = ST_MENU;
            state = ST_MENU;
            break;
        case ACT_QUIT: quit = true; break;
        default: break;
        }
    }

    UiUnload();
    GfxUnload();
    AudioShutdown();
    CloseWindow();
    return 0;
}
