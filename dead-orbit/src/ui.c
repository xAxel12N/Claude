// Menus, screens and 2D drawing helpers.
#include "ui.h"
#include "audio.h"
#include "settings.h"
#include "raymath.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

Font fontTitle, fontUI, fontMono;
static bool ownFonts;

static const Color ACCENT = { 80, 230, 255, 255 };
static const Color DIM = { 150, 170, 190, 255 };

static Font LoadGameFont(const char *file, int size)
{
    const char *dirs[3] = {
        TextFormat("%sassets/fonts/%s", GetApplicationDirectory(), file),
        TextFormat("%s../share/dead-orbit/assets/fonts/%s", GetApplicationDirectory(), file),
        TextFormat("assets/fonts/%s", file),
    };
    for (int i = 0; i < 3; i++) {
        char path[1024];
        snprintf(path, sizeof path, "%s", dirs[i]);
        if (FileExists(path)) {
            Font f = LoadFontEx(path, size, NULL, 0);
            if (f.texture.id) {
                GenTextureMipmaps(&f.texture);
                SetTextureFilter(f.texture, TEXTURE_FILTER_TRILINEAR);
                return f;
            }
        }
    }
    return GetFontDefault();
}

void UiInit(void)
{
    fontTitle = LoadGameFont("DejaVuSans-Bold.ttf", 128);
    fontUI = LoadGameFont("DejaVuSans-Bold.ttf", 48);
    fontMono = LoadGameFont("DejaVuSansMono-Bold.ttf", 40);
    ownFonts = fontTitle.texture.id != GetFontDefault().texture.id;
}

void UiUnload(void)
{
    if (!ownFonts) return;
    UnloadFont(fontTitle);
    UnloadFont(fontUI);
    UnloadFont(fontMono);
}

Vector2 UiMeasure(Font f, const char *s, float size) { return MeasureTextEx(f, s, size, size*0.02f); }

void UiText(Font f, const char *s, float x, float y, float size, Color c)
{
    Color sh = { 0, 0, 0, (unsigned char)(c.a*0.6f) };
    DrawTextEx(f, s, (Vector2){ x + size*0.05f, y + size*0.06f }, size, size*0.02f, sh);
    DrawTextEx(f, s, (Vector2){ x, y }, size, size*0.02f, c);
}

void UiTextCentered(Font f, const char *s, float cx, float y, float size, Color c)
{
    Vector2 m = UiMeasure(f, s, size);
    UiText(f, s, cx - m.x/2, y, size, c);
}

void UiPanel(Rectangle r, Color fill, Color edge)
{
    DrawRectangleRounded(r, 0.12f, 8, fill);
    DrawRectangleRoundedLinesEx(r, 0.12f, 8, 1.5f, edge);
}

void UiBar(Rectangle r, float frac, Color fill, Color back)
{
    frac = Clamp(frac, 0, 1);
    DrawRectangleRounded(r, 0.5f, 6, back);
    if (frac > 0.01f) DrawRectangleRounded((Rectangle){ r.x, r.y, r.width*frac, r.height }, 0.5f, 6, fill);
    for (int i = 1; i < 10; i++) {
        float x = r.x + r.width*i/10.0f;
        DrawLineEx((Vector2){ x, r.y + 2 }, (Vector2){ x, r.y + r.height - 2 }, 1, (Color){ 0, 0, 0, 90 });
    }
}

// Buddy, the ship AI: a boxy little robot face that yaps.
void UiDrawBuddy(Vector2 c, float s, float talk, float t)
{
    if (s < 1) return;
    Rectangle head = { c.x - s, c.y - s*0.8f, s*2, s*1.6f };
    DrawLineEx((Vector2){ c.x + s*0.4f, head.y }, (Vector2){ c.x + s*0.6f, head.y - s*0.5f }, s*0.08f, (Color){ 160, 170, 180, 255 });
    DrawCircleV((Vector2){ c.x + s*0.6f, head.y - s*0.5f }, s*0.14f, fmodf(t, 1) < 0.5f ? (Color){ 255, 60, 50, 255 } : (Color){ 120, 30, 30, 255 });
    DrawRectangleRounded(head, 0.35f, 8, (Color){ 120, 200, 220, 255 });
    DrawRectangleRounded((Rectangle){ head.x + s*0.15f, head.y + s*0.15f, head.width - s*0.3f, head.height - s*0.3f }, 0.3f, 8, (Color){ 20, 40, 60, 255 });
    bool blink = fmodf(t, 3.3f) < 0.12f;
    for (int i = -1; i <= 1; i += 2) {
        Vector2 e = { c.x + i*s*0.42f, c.y - s*0.15f };
        float er = s*(i < 0 ? 0.3f : 0.24f);
        if (blink) DrawRectangle((int)(e.x - er), (int)e.y, (int)(er*2), (int)(s*0.06f) + 1, ACCENT);
        else {
            DrawCircleV(e, er, (Color){ 220, 250, 255, 255 });
            DrawCircleV((Vector2){ e.x + sinf(t*1.7f)*er*0.3f, e.y + cosf(t*2.3f)*er*0.2f }, er*0.45f, (Color){ 10, 20, 30, 255 });
        }
    }
    float mh = talk > 0 ? s*(0.08f + 0.22f*fabsf(sinf(t*17))) : s*0.06f;
    DrawRectangleRounded((Rectangle){ c.x - s*0.4f, c.y + s*0.32f - mh/2, s*0.8f, mh }, 0.5f, 4, ACCENT);
}

void UiDrawGlooper2D(Vector2 c, float r, Vector2 look, float t, Color col)
{
    float w = sinf(t*5)*0.06f;
    for (int a = 0; a < 5; a++) {
        float ang = PI*0.15f + a*PI*0.175f;
        for (int j = 1; j <= 6; j++) {
            float sw = sinf(t*4 + j*0.7f + a)*0.12f*j*r*0.4f;
            Vector2 p = { c.x + cosf(ang)*r*(0.5f + j*0.1f) + sw, c.y + sinf(ang)*r*(0.55f + j*0.22f) };
            DrawCircleV(p, r*0.2f*(1 - j/7.5f), (Color){ (unsigned char)(col.r*0.8f), (unsigned char)(col.g*0.8f), (unsigned char)(col.b*0.8f), 255 });
        }
    }
    DrawEllipse((int)c.x, (int)c.y, r*(1 + w), r*(1 - w), col);
    DrawCircleV((Vector2){ c.x - r*0.35f, c.y - r*0.4f }, r*0.22f, (Color){ 255, 255, 255, 70 });
    Vector2 eyes[2] = { { c.x - r*0.38f, c.y - r*0.25f }, { c.x + r*0.4f, c.y - r*0.32f } };
    float er[2] = { r*0.36f, r*0.26f };
    for (int i = 0; i < 2; i++) {
        DrawCircleV(eyes[i], er[i], (Color){ 248, 248, 240, 255 });
        Vector2 d = Vector2Subtract(look, eyes[i]);
        float l = Vector2Length(d);
        if (l > 1) d = Vector2Scale(d, 1/l);
        Vector2 jig = { sinf(t*13 + i*2)*0.15f, cosf(t*11 + i)*0.15f };
        Vector2 pc = Vector2Add(eyes[i], Vector2Scale(Vector2Add(d, jig), er[i]*0.5f));
        DrawCircleV(pc, er[i]*0.45f, (Color){ 10, 10, 12, 255 });
    }
    Vector2 m = { c.x, c.y + r*0.32f };
    DrawEllipse((int)m.x, (int)m.y, r*0.55f, r*0.22f, (Color){ 45, 6, 18, 255 });
    for (int k = 0; k < 6; k++) {
        float x = m.x - r*0.42f + k*r*0.17f;
        float y = m.y - r*0.17f;
        DrawTriangle((Vector2){ x - r*0.06f, y }, (Vector2){ x, y + r*0.15f }, (Vector2){ x + r*0.06f, y }, (Color){ 250, 248, 230, 255 });
    }
    DrawEllipse((int)(m.x + r*0.18f), (int)(m.y + r*0.2f), r*0.13f, r*0.09f, (Color){ 255, 110, 150, 255 });
}

static void DrawMimic2D(Vector2 c, float s, float t)
{
    float open = 0.35f + 0.25f*sinf(t*8);
    DrawRectangleRec((Rectangle){ c.x - s, c.y, s*2, s }, (Color){ 150, 96, 46, 255 });
    DrawRectangleRec((Rectangle){ c.x - s, c.y + s*0.4f, s*2, s*0.15f }, (Color){ 70, 70, 76, 255 });
    DrawRectangleRec((Rectangle){ c.x - s*0.9f, c.y - s*0.05f, s*1.8f, s*0.1f }, (Color){ 90, 10, 25, 255 });
    for (int k = 0; k < 6; k++) {
        float x = c.x - s*0.8f + k*s*0.32f;
        DrawTriangle((Vector2){ x + s*0.12f, c.y }, (Vector2){ x, c.y }, (Vector2){ x + s*0.06f, c.y - s*0.22f }, (Color){ 250, 248, 230, 255 });
    }
    Vector2 hinge = { c.x - s, c.y };
    float a = -open;
    Vector2 p1 = { hinge.x + cosf(a)*s*2, hinge.y + sinf(a)*s*2 };
    Vector2 up = { sinf(a)*s*0.35f, -cosf(a)*s*0.35f };
    DrawTriangle(hinge, p1, Vector2Add(p1, up), (Color){ 150, 96, 46, 255 });
    DrawTriangle(hinge, Vector2Add(p1, up), Vector2Add(hinge, up), (Color){ 150, 96, 46, 255 });
    DrawEllipse((int)(c.x + s*0.4f), (int)(c.y + s*0.05f), s*0.5f, s*0.14f, (Color){ 255, 105, 150, 255 });
    for (int i = 0; i < 2; i++) {
        Vector2 root = Vector2Lerp(hinge, p1, 0.35f + i*0.35f);
        Vector2 tip = { root.x + sinf(t*3 + i)*s*0.2f, root.y - s*0.9f };
        DrawLineEx(root, tip, s*0.08f, (Color){ 255, 120, 160, 255 });
        DrawCircleV(tip, s*(i ? 0.22f : 0.28f), (Color){ 248, 248, 240, 255 });
        DrawCircleV((Vector2){ tip.x + sinf(t*9 + i)*s*0.06f, tip.y + s*0.05f }, s*0.11f, BLACK);
    }
}

// ------------------------------------------------------------------ list menu widget
typedef struct { const char *label; const char *value; } Entry;

static float Scale(void) { float s = GetScreenHeight()/900.0f; return s < 0.6f ? 0.6f : s; }

// Returns the activated index (or -1). *lr receives -1/+1 when a value entry is changed.
static int ListMenu(const Entry *e, int n, float x, float y, float w, int *sel, int *lr)
{
    float S = Scale();
    float h = 54*S, gap = 10*S;
    int activated = -1;
    *lr = 0;
    if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) { *sel = (*sel + 1) % n; SfxPlay(SFX_CLICK, 0.5f, 1); }
    if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) { *sel = (*sel + n - 1) % n; SfxPlay(SFX_CLICK, 0.5f, 1); }
    Vector2 mouse = GetMousePosition();
    static Vector2 lastMouse;
    bool mouseMoved = Vector2Distance(mouse, lastMouse) > 0.5f;
    lastMouse = mouse;
    for (int i = 0; i < n; i++) {
        Rectangle r = { x, y + i*(h + gap), w, h };
        if (CheckCollisionPointRec(mouse, r)) {
            if (mouseMoved && *sel != i) { *sel = i; SfxPlay(SFX_CLICK, 0.4f, 1.1f); }
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                *sel = i;
                if (e[i].value) *lr = mouse.x < r.x + r.width*0.5f ? -1 : 1;
                else activated = i;
            }
        }
    }
    if (e[*sel].value) {
        if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) *lr = -1;
        if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D) || IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_SPACE)) *lr = 1;
    } else if (IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_SPACE)) activated = *sel;
    if (*lr) SfxPlay(SFX_CLICK, 0.6f, *lr > 0 ? 1.2f : 0.9f);
    if (activated >= 0) SfxPlay(SFX_SELECT, 0.7f, 1);

    float t = (float)GetTime();
    for (int i = 0; i < n; i++) {
        Rectangle r = { x, y + i*(h + gap), w, h };
        bool s = i == *sel;
        float pulse = s ? 0.5f + 0.5f*sinf(t*5) : 0;
        DrawRectangleRounded(r, 0.25f, 8, s ? (Color){ 20, 60, 80, 220 } : (Color){ 8, 14, 24, 170 });
        DrawRectangleRoundedLinesEx(r, 0.25f, 8, s ? 2.0f : 1.0f, s ? Fade(ACCENT, 0.7f + 0.3f*pulse) : (Color){ 60, 80, 100, 160 });
        if (s) DrawRectangleRounded((Rectangle){ r.x + 8*S, r.y + 12*S, 5*S, r.height - 24*S }, 1, 4, ACCENT);
        UiText(fontUI, e[i].label, r.x + 26*S, r.y + (h - 26*S)/2, 26*S, s ? WHITE : DIM);
        if (e[i].value) {
            char v[64];
            snprintf(v, sizeof v, "<  %s  >", e[i].value);
            Vector2 m = UiMeasure(fontUI, v, 24*S);
            UiText(fontUI, v, r.x + r.width - m.x - 22*S, r.y + (h - 24*S)/2, 24*S, s ? ACCENT : DIM);
        }
    }
    return activated;
}

static void Backdrop(float alpha)
{
    int sw = GetScreenWidth(), sh = GetScreenHeight();
    DrawRectangleGradientH(0, 0, sw*0.62f, sh, (Color){ 0, 4, 10, (unsigned char)(220*alpha) }, (Color){ 0, 4, 10, 0 });
}

static void Footer(const char *text)
{
    float S = Scale();
    UiText(fontMono, text, 40*S, GetScreenHeight() - 44*S, 16*S, Fade(DIM, 0.8f));
}

static void Title(float x, float y, float size, float t)
{
    const char *a = "DEAD", *b = "ORBIT";
    float flick = (fmodf(t*7.3f, 13.0f) < 0.4f) ? 0.35f : 1.0f;
    for (int g = 4; g >= 1; g--) {
        Color glow = Fade((Color){ 255, 60, 50, 255 }, 0.07f*flick);
        DrawTextEx(fontTitle, a, (Vector2){ x - g*2.0f, y - g*1.0f }, size + g*3, size*0.04f, glow);
    }
    DrawTextEx(fontTitle, a, (Vector2){ x + 4, y + 5 }, size, size*0.04f, (Color){ 0, 0, 0, 160 });
    DrawTextEx(fontTitle, a, (Vector2){ x, y }, size, size*0.04f, Fade((Color){ 255, 72, 60, 255 }, flick));
    Vector2 m = MeasureTextEx(fontTitle, a, size, size*0.04f);
    float bx = x + m.x + size*0.22f;
    DrawTextEx(fontTitle, b, (Vector2){ bx + 4, y + 5 }, size, size*0.04f, (Color){ 0, 0, 0, 160 });
    DrawTextEx(fontTitle, b, (Vector2){ bx, y }, size, size*0.04f, (Color){ 235, 240, 245, 255 });
    // googly eyes peek out of the O
    Vector2 om = MeasureTextEx(fontTitle, "O", size, size*0.04f);
    Vector2 oc = { bx + om.x*0.5f, y + size*0.5f };
    Vector2 mouse = GetMousePosition();
    for (int i = -1; i <= 1; i += 2) {
        Vector2 e = { oc.x + i*om.x*0.16f, oc.y };
        float er = om.x*(i < 0 ? 0.13f : 0.1f);
        DrawCircleV(e, er, WHITE);
        Vector2 d = Vector2Subtract(mouse, e);
        float l = Vector2Length(d);
        if (l > 1) d = Vector2Scale(d, 1/l);
        DrawCircleV(Vector2Add(e, Vector2Scale(d, er*0.5f)), er*0.5f, BLACK);
    }
}

// ------------------------------------------------------------------ screens
UiAction UiMainMenu(float t)
{
    static int sel;
    float S = Scale();
    int sw = GetScreenWidth(), sh = GetScreenHeight();
    Backdrop(1);
    float x = 80*S;
    Title(x, 70*S, 120*S, t);
    UiText(fontUI, "a zero-g salvage horror-comedy", x + 6*S, 205*S, 26*S, (Color){ 200, 215, 230, 230 });

    static const char *diffNames[DIFF_COUNT] = { "EASY", "NORMAL", "NIGHTMARE" };
    Entry e[] = {
        { "START SHIFT", NULL },
        { "DIFFICULTY", diffNames[settings.difficulty] },
        { "HOW TO PLAY", NULL },
        { "SETTINGS", NULL },
        { "QUIT", NULL },
    };
    int lr;
    int act = ListMenu(e, 5, x, 290*S, 470*S, &sel, &lr);
    if (lr && sel == 1) {
        settings.difficulty = (settings.difficulty + lr + DIFF_COUNT) % DIFF_COUNT;
        SettingsSave();
    }
    static const char *diffInfo[DIFF_COUNT] = {
        "Quota $500. Big air reserve. Two sleepy Gloopers.",
        "Quota $900. The intended experience.",
        "Quota $1300. Seven hungry Gloopers. Most crates bite.",
    };
    UiText(fontMono, diffInfo[settings.difficulty], x + 4*S, 290*S + 5*64*S + 10*S, 17*S, Fade(ACCENT, 0.85f));
    char best[64];
    snprintf(best, sizeof best, "BEST %s: %d", diffNames[settings.difficulty], settings.best[settings.difficulty]);
    UiText(fontMono, best, x + 4*S, 290*S + 5*64*S + 38*S, 17*S, (Color){ 255, 210, 90, 230 });

    // a glooper peeks in from the corner and watches the mouse
    float pk = 0.5f + 0.5f*sinf(t*0.6f);
    Vector2 gc = { sw - 140*S, sh - 60*S + (1 - pk)*90*S };
    UiDrawGlooper2D(gc, 95*S, GetMousePosition(), t, (Color){ 120, 235, 90, 255 });
    Footer("Dead Orbit v1.0   offline single-player   mouse + keyboard");
    if (IsKeyPressed(KEY_ESCAPE)) return ACT_QUIT;
    switch (act) {
    case 0: return ACT_START;
    case 2: return ACT_HOWTO;
    case 3: return ACT_SETTINGS;
    case 4: return ACT_QUIT;
    }
    return ACT_NONE;
}

UiAction UiSettingsMenu(float t, bool inGame)
{
    static int sel;
    float S = Scale();
    Backdrop(1);
    if (inGame) DrawRectangle(0, 0, GetScreenWidth(), GetScreenHeight(), (Color){ 0, 0, 0, 120 });
    float x = 80*S;
    UiText(fontTitle, "SETTINGS", x, 70*S, 72*S, WHITE);
    char sens[16], mv[16], mu[16];
    snprintf(sens, sizeof sens, "%.1f", settings.mouseSens);
    snprintf(mv, sizeof mv, "%d%%", (int)roundf(settings.masterVol*100));
    snprintf(mu, sizeof mu, "%d%%", (int)roundf(settings.musicVol*100));
    Entry e[] = {
        { "MOUSE SENSITIVITY", sens },
        { "INVERT MOUSE Y", settings.invertY ? "ON" : "OFF" },
        { "MASTER VOLUME", mv },
        { "MUSIC VOLUME", mu },
        { "FULLSCREEN", settings.fullscreen ? "ON" : "OFF" },
        { "BLOOM GLOW", settings.bloom ? "ON" : "OFF" },
        { "SCREEN SHAKE", settings.screenShake ? "ON" : "OFF" },
        { "BACK", NULL },
    };
    int lr;
    int act = ListMenu(e, 8, x, 190*S, 560*S, &sel, &lr);
    if (lr) {
        switch (sel) {
        case 0: settings.mouseSens = Clamp(roundf((settings.mouseSens + lr*0.1f)*10)/10, 0.2f, 3.0f); break;
        case 1: settings.invertY = !settings.invertY; break;
        case 2: settings.masterVol = Clamp(settings.masterVol + lr*0.1f, 0, 1); break;
        case 3: settings.musicVol = Clamp(settings.musicVol + lr*0.1f, 0, 1); break;
        case 4: settings.fullscreen = !settings.fullscreen; break;
        case 5: settings.bloom = !settings.bloom; break;
        case 6: settings.screenShake = !settings.screenShake; break;
        }
        AudioSetVolumes(settings.masterVol, settings.musicVol);
    }
    Footer("LEFT/RIGHT or click to change   ESC to go back");
    if (act == 7 || IsKeyPressed(KEY_ESCAPE)) { SettingsSave(); return ACT_BACK; }
    return ACT_NONE;
}

UiAction UiHowTo(float t)
{
    float S = Scale();
    int sw = GetScreenWidth(), sh = GetScreenHeight();
    DrawRectangle(0, 0, sw, sh, (Color){ 0, 4, 10, 200 });
    float x = 80*S;
    UiText(fontTitle, "HOW TO PLAY", x, 50*S, 64*S, WHITE);
    UiText(fontUI, "Salvage the wrecked station, haul the loot to your tug's green bay, hit the quota, go home.", x, 130*S, 21*S, (Color){ 210, 225, 240, 255 });
    UiText(fontUI, "Your suit leaks oxygen every second. Thrusting, boosting and torching burn even more.", x, 160*S, 21*S, (Color){ 210, 225, 240, 255 });

    const char *keys[][2] = {
        { "MOUSE", "look around" },
        { "W A S D", "thrust" },
        { "SPACE / CTRL", "thrust up / down" },
        { "SHIFT", "boost (mash it to shake off huggers)" },
        { "X", "brake" },
        { "LEFT MOUSE", "plasma torch: cut bolts, pop Gloopers" },
        { "RIGHT MOUSE", "fire a tether at loose salvage (3 max)" },
        { "R  (hold)", "reel tethers in" },
        { "E", "release all tethers" },
        { "ESC", "pause" },
    };
    Rectangle kp = { x, 205*S, 600*S, 470*S };
    UiPanel(kp, (Color){ 6, 12, 22, 200 }, (Color){ 60, 90, 120, 200 });
    UiText(fontUI, "CONTROLS", kp.x + 22*S, kp.y + 16*S, 24*S, ACCENT);
    for (int i = 0; i < 10; i++) {
        UiText(fontMono, keys[i][0], kp.x + 22*S, kp.y + 60*S + i*39*S, 19*S, (Color){ 255, 210, 90, 255 });
        UiText(fontUI, keys[i][1], kp.x + 210*S, kp.y + 60*S + i*39*S, 19*S, (Color){ 220, 230, 240, 255 });
    }

    Rectangle lp = { x + 630*S, 205*S, sw - x - 630*S - 80*S, 470*S };
    if (lp.width > 300*S) {
        UiPanel(lp, (Color){ 6, 12, 22, 200 }, (Color){ 60, 90, 120, 200 });
        UiText(fontUI, "MEET THE LOCALS", lp.x + 22*S, lp.y + 16*S, 24*S, ACCENT);
        float rowH = 135*S;
        Vector2 icon = { lp.x + 70*S, lp.y + 110*S };
        UiDrawGlooper2D(icon, 38*S, (Vector2){ icon.x + 50, icon.y }, t, (Color){ 190, 110, 245, 255 });
        UiText(fontUI, "GLOOPER", lp.x + 140*S, lp.y + 70*S, 22*S, (Color){ 190, 110, 245, 255 });
        UiText(fontUI, "Wants a hug. Hugs drain your air fast.", lp.x + 140*S, lp.y + 100*S, 18*S, WHITE);
        UiText(fontUI, "Torch it or mash SHIFT to fling it off.", lp.x + 140*S, lp.y + 126*S, 18*S, WHITE);
        DrawMimic2D((Vector2){ icon.x, icon.y + rowH }, 34*S, t);
        UiText(fontUI, "MIMIC", lp.x + 140*S, lp.y + 70*S + rowH, 22*S, (Color){ 255, 120, 160, 255 });
        UiText(fontUI, "Some crates are not crates. They bite.", lp.x + 140*S, lp.y + 100*S + rowH, 18*S, WHITE);
        UiText(fontUI, "Torch one and its tongue is worth $150.", lp.x + 140*S, lp.y + 126*S + rowH, 18*S, WHITE);
        UiDrawBuddy((Vector2){ icon.x, icon.y + rowH*2 + 10*S }, 30*S, fmodf(t, 2) < 1, t);
        UiText(fontUI, "BUDDY", lp.x + 140*S, lp.y + 70*S + rowH*2, 22*S, ACCENT);
        UiText(fontUI, "Your ship's AI. Helpful. Mostly.", lp.x + 140*S, lp.y + 100*S + rowH*2, 18*S, WHITE);
        UiText(fontUI, "Stand in the green bay to refill your O2.", lp.x + 140*S, lp.y + 126*S + rowH*2, 18*S, WHITE);
    }
    UiText(fontUI, "Bolted loot needs cutting first. Blue canisters give +30 O2. Heavy loot pulls you around.", x, 700*S, 19*S, (Color){ 255, 210, 90, 255 });
    Footer("Press ESC, ENTER or click to go back");
    if (IsKeyPressed(KEY_ESCAPE) || IsKeyPressed(KEY_ENTER) || IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        SfxPlay(SFX_SELECT, 0.6f, 0.9f);
        return ACT_BACK;
    }
    return ACT_NONE;
}

UiAction UiPauseMenu(float t)
{
    static int sel;
    float S = Scale();
    DrawRectangle(0, 0, GetScreenWidth(), GetScreenHeight(), (Color){ 0, 4, 10, 150 });
    Backdrop(0.8f);
    float x = 80*S;
    UiText(fontTitle, "PAUSED", x, 120*S, 72*S, WHITE);
    UiText(fontUI, "The station will wait. Your oxygen will too.", x + 4*S, 205*S, 22*S, DIM);
    Entry e[] = { { "RESUME", NULL }, { "HOW TO PLAY", NULL }, { "SETTINGS", NULL }, { "ABANDON SHIFT", NULL } };
    int lr;
    int act = ListMenu(e, 4, x, 280*S, 440*S, &sel, &lr);
    if (IsKeyPressed(KEY_ESCAPE)) return ACT_RESUME;
    switch (act) {
    case 0: return ACT_RESUME;
    case 1: return ACT_HOWTO;
    case 2: return ACT_SETTINGS;
    case 3: sel = 0; return ACT_ABANDON;
    }
    return ACT_NONE;
}

UiAction UiEndScreen(float t, const RunStats *s, bool won, bool newBest)
{
    static int sel;
    float S = Scale();
    int sw = GetScreenWidth(), sh = GetScreenHeight();
    DrawRectangle(0, 0, sw, sh, (Color){ 0, 4, 10, 190 });
    float cx = sw/2.0f;
    static const char *deadSubs[] = {
        "You are now a very cold, very expensive satellite.",
        "The Gloopers will remember you fondly.",
        "Buddy has already listed your bunk for rent.",
    };
    if (won) {
        UiTextCentered(fontTitle, "SHIFT COMPLETE", cx, 110*S, 80*S, (Color){ 90, 255, 150, 255 });
        UiTextCentered(fontUI, "You made quota and nobody got eaten. Probably.", cx, 210*S, 24*S, (Color){ 210, 225, 240, 255 });
    } else {
        UiTextCentered(fontTitle, "SPACE POPSICLE", cx, 110*S, 80*S, (Color){ 255, 80, 70, 255 });
        UiTextCentered(fontUI, deadSubs[(int)(s->time) % 3], cx, 210*S, 24*S, (Color){ 210, 225, 240, 255 });
        if (s->deathCause) {
            char c[96];
            snprintf(c, sizeof c, "Cause of death: %s", s->deathCause);
            UiTextCentered(fontMono, c, cx, 248*S, 19*S, (Color){ 255, 150, 130, 255 });
        }
    }
    Rectangle p = { cx - 300*S, 300*S, 600*S, 250*S };
    UiPanel(p, (Color){ 6, 12, 22, 210 }, (Color){ 60, 90, 120, 200 });
    char line[96];
    const char *labels[5] = { "SALVAGE", "ITEMS HAULED", "GLOOPERS POPPED", "MIMICS TORCHED", "TIME" };
    char vals[5][32];
    snprintf(vals[0], 32, "$%d / $%d", s->credits, s->quota);
    snprintf(vals[1], 32, "%d", s->deposited);
    snprintf(vals[2], 32, "%d", s->gloopsPopped);
    snprintf(vals[3], 32, "%d", s->mimicsSlain);
    snprintf(vals[4], 32, "%d:%02d", (int)s->time/60, (int)s->time % 60);
    for (int i = 0; i < 5; i++) {
        UiText(fontMono, labels[i], p.x + 30*S, p.y + 24*S + i*34*S, 19*S, DIM);
        Vector2 m = UiMeasure(fontMono, vals[i], 19*S);
        UiText(fontMono, vals[i], p.x + p.width - 30*S - m.x, p.y + 24*S + i*34*S, 19*S, WHITE);
    }
    if (won) {
        snprintf(line, sizeof line, "SCORE %d%s", s->score, newBest ? "   NEW BEST!" : "");
        UiTextCentered(fontTitle, line, cx, p.y + 196*S, 32*S, newBest ? (Color){ 255, 210, 90, 255 } : WHITE);
    }
    Entry e[] = { { won ? "NEW SHIFT" : "TRY AGAIN", NULL }, { "MAIN MENU", NULL } };
    int lr;
    int act = ListMenu(e, 2, cx - 220*S, 590*S, 440*S, &sel, &lr);
    if (act == 0) return ACT_RETRY;
    if (act == 1 || IsKeyPressed(KEY_ESCAPE)) { sel = 0; return ACT_MENU; }
    return ACT_NONE;
}
