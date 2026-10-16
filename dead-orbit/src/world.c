// Dead Orbit gameplay: station, player, salvage, tethers, creatures, ship and HUD.
#include "world.h"
#include "gfx.h"
#include "audio.h"
#include "ui.h"
#include "rlgl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PARTS 400
#define MAX_COLLIDERS 160
#define MAX_ITEMS 48
#define MAX_GLOOPERS 10
#define MAX_TETHERS 3
#define PLAYER_R 0.8f
#define CUT_RANGE 14.0f
#define TETHER_RANGE 20.0f
#define BAY_R 4.6f

static const Vector3 SHIP_POS = { 0.0f, 3.0f, -58.0f };
static const Vector3 BAY_POS = { 0.0f, 3.0f, -50.5f };
static const Vector3 BREACH_LIGHT = { 10.0f, 3.6f, 4.0f };
static const Vector3 ANTENNA_TIP = { 0.0f, 9.2f, 0.0f };

typedef struct {
    float o2Drain, reserve;
    int quota, gloopers;
    float gloopSpeed, gloopSense, gloopDrain, mimicChance;
} DiffParams;

static const DiffParams DIFFS[DIFF_COUNT] = {
    { 0.55f, 300, 500, 2, 3.4f, 22, 3.0f, 0.25f },
    { 0.85f, 200, 900, 4, 4.4f, 30, 4.5f, 0.40f },
    { 1.20f, 120, 1300, 7, 5.8f, 42, 6.5f, 0.60f },
};

// ------------------------------------------------------------------ types
typedef struct { MeshKind mesh; Matrix m; Color c; float emissive; Pattern pat; float gloss; int blink; } Part;
typedef enum { COL_SPHERE, COL_CAPSULE, COL_BOX } ColKind;
typedef struct { ColKind kind; Vector3 a, b; float r; Matrix inv, rot; Vector3 half; } Collider;

typedef enum { IT_SCRAP, IT_BATTERY, IT_CORE, IT_TOILET, IT_BOX, IT_O2, IT_TONGUE } ItemType;
typedef struct {
    bool active; ItemType type;
    Vector3 pos, vel, rot, spin;
    float radius, mass; int value;
    bool bolted; float cut;
    bool mimic, awake; float hp, biteCd, hopT, chomp;
    int tether;
} Item;

typedef struct { bool active; int item; float rest; } Tether;

typedef enum { G_LURK, G_CHASE, G_LATCHED, G_STUNNED, G_DEAD } GState;
typedef struct {
    GState state; Vector3 pos, vel, home, latchOff;
    float phase, size, hp, timer, giggleT; Color col;
} Glooper;

typedef struct {
    Vector3 pos, vel;
    float yaw, pitch;
    Vector3 bodyFwd, bodyUp;
    float o2, hurt, shake;
    int shakeCount;
    bool cutting, cutHit; Vector3 cutEnd;
    float thrustVis, lean, strafe;
} Player;

typedef enum { TGT_NONE, TGT_STATION, TGT_ITEM, TGT_GLOOP } TargetKind;
typedef struct { TargetKind kind; int index; float dist; Vector3 point; } Target;

// ------------------------------------------------------------------ state
static Part parts[MAX_PARTS]; static int partCount;
static Collider cols[MAX_COLLIDERS]; static int colCount;
static Item items[MAX_ITEMS]; static int itemCount;
static Glooper gloops[MAX_GLOOPERS]; static int gloopCount;
static Tether tethers[MAX_TETHERS];
static Player pl;
static Target target;
static RunStats stats;
static DiffParams dp;
static Difficulty diff;
static Camera3D cam;
static Vector3 camPosSmooth;
static float worldTime, deadT, winT, camShake;
static bool dead, won, quotaAnnounced;
static float boltHintCd, farCd, bumpCd, nearGloopCd;
static bool saidLow, saidCrit, saidReserve, saidGloop;

static const Vector3 VENTS[] = {
    { -10, -2.8f, 5 }, { 10, -2.8f, -5 }, { 0, -4.8f, 0 }, { -26, 1.8f, 0 }, { 26, -1.8f, 0 }, { 0, 0, -12.5f },
};
#define VENT_COUNT ((int)(sizeof VENTS/sizeof VENTS[0]))

// radio chatter from Buddy, the ship's AI
static char radioText[200];
static float radioT = 99, radioDur = 1;

static float Rf(float a, float b) { return a + (b - a)*(rand()/(float)RAND_MAX); }
static Vector3 V3(float x, float y, float z) { return (Vector3){ x, y, z }; }
static Color Col(int r, int g, int b) { return (Color){ (unsigned char)r, (unsigned char)g, (unsigned char)b, 255 }; }

static void Radio(const char *msg)
{
    snprintf(radioText, sizeof radioText, "%s", msg);
    radioT = 0;
    radioDur = 3.5f + strlen(msg)/28.0f;
    SfxPlay(SFX_RADIO, 0.6f, 1.0f);
}

static void RadioPick(const char **pool, int n) { Radio(pool[rand() % n]); }

// ------------------------------------------------------------------ station construction
static void AddPart(MeshKind mesh, Matrix m, Color c, float em, Pattern pat, float gloss, int blink)
{
    if (partCount < MAX_PARTS) parts[partCount++] = (Part){ mesh, m, c, em, pat, gloss, blink };
}

static Matrix SegMatrix(Vector3 a, Vector3 b, float r)
{
    Vector3 d = Vector3Subtract(b, a);
    float len = Vector3Length(d);
    Vector3 up = Vector3Scale(d, 1.0f/len);
    Vector3 ref = fabsf(up.y) < 0.95f ? V3(0, 1, 0) : V3(1, 0, 0);
    Vector3 right = Vector3Normalize(Vector3CrossProduct(ref, up));
    Vector3 fwd = Vector3CrossProduct(right, up);
    return MatBasis(Vector3Scale(right, r), Vector3Scale(up, len), Vector3Scale(fwd, r), a);
}

static void AddSeg(Vector3 a, Vector3 b, float r, Color c, float em, Pattern pat, float gloss)
{
    AddPart(MESH_CYL, SegMatrix(a, b, r), c, em, pat, gloss, 0);
}

static void ColSphere(Vector3 c, float r)
{
    if (colCount < MAX_COLLIDERS) cols[colCount++] = (Collider){ .kind = COL_SPHERE, .a = c, .r = r };
}

static void ColCapsule(Vector3 a, Vector3 b, float r)
{
    if (colCount < MAX_COLLIDERS) cols[colCount++] = (Collider){ .kind = COL_CAPSULE, .a = a, .b = b, .r = r };
}

static void ColBox(Vector3 c, Vector3 e, Vector3 size)
{
    if (colCount >= MAX_COLLIDERS) return;
    Matrix rot = MatrixRotateXYZ(e);
    Collider k = { .kind = COL_BOX, .a = c, .rot = rot, .half = Vector3Scale(size, 0.5f) };
    k.inv = MatrixInvert(MatrixMultiply(rot, MatrixTranslate(c.x, c.y, c.z)));
    cols[colCount++] = k;
}

static void AddBox(Vector3 c, Vector3 e, Vector3 size, Color col, float em, Pattern pat, float gloss, bool collide)
{
    AddPart(MESH_CUBE, MatTRS(c, e, size), col, em, pat, gloss, 0);
    if (collide) ColBox(c, e, size);
}

static void BuildStation(void)
{
    partCount = 0; colCount = 0;
    Color hull = Col(178, 180, 184), dark = Col(70, 72, 78), white = Col(206, 206, 200), gold = Col(222, 165, 70);

    // central hub
    AddPart(MESH_SPHERE, MatTRS(V3(0, 0, 0), V3(0, 0, 0), V3(4.2f, 4.2f, 4.2f)), hull, 0, PAT_PANEL, 0.35f, 0);
    ColSphere(V3(0, 0, 0), 4.2f);
    AddPart(MESH_TORUS, MatTRS(V3(0, 0, 0), V3(PI/2, 0, 0), V3(4.4f, 4.4f, 4.4f)), dark, 0, PAT_PANEL, 0.4f, 0);

    // main truss with lattice rails
    AddSeg(V3(-26, 0, 0), V3(26, 0, 0), 0.6f, hull, 0, PAT_PANEL, 0.3f);
    for (int k = 0; k < 4; k++) {
        float y = (k & 1) ? 1.1f : -1.1f, z = (k & 2) ? 1.1f : -1.1f;
        AddSeg(V3(-26, y, z), V3(26, y, z), 0.11f, dark, 0, PAT_NONE, 0.5f);
    }
    for (int i = 0; i < 26; i++) {
        float x0 = -26 + i*2.0f, x1 = x0 + 2.0f;
        if (fabsf(x0) < 4.5f) continue;
        int s = i & 1;
        AddSeg(V3(x0, s ? 1.1f : -1.1f, 1.1f), V3(x1, s ? -1.1f : 1.1f, 1.1f), 0.07f, dark, 0, PAT_NONE, 0.5f);
        AddSeg(V3(x0, s ? 1.1f : -1.1f, -1.1f), V3(x1, s ? -1.1f : 1.1f, -1.1f), 0.07f, dark, 0, PAT_NONE, 0.5f);
        AddSeg(V3(x0, 1.1f, s ? 1.1f : -1.1f), V3(x1, 1.1f, s ? -1.1f : 1.1f), 0.07f, dark, 0, PAT_NONE, 0.5f);
    }
    ColCapsule(V3(-26, 0, 0), V3(26, 0, 0), 1.6f);

    // habitat modules: white (left) and gold foil (right, breached)
    for (int s = -1; s <= 1; s += 2) {
        Color c = s < 0 ? white : gold;
        Pattern p = s < 0 ? PAT_PANEL : PAT_FOIL;
        AddSeg(V3(s*10.0f, 0, -8), V3(s*10.0f, 0, 8), 2.4f, c, 0, p, s < 0 ? 0.35f : 0.8f);
        for (int e = -1; e <= 1; e += 2)
            AddPart(MESH_SPHERE, MatTRS(V3(s*10.0f, 0, e*8.0f), V3(0, 0, 0), V3(2.4f, 2.4f, 1.8f)), c, 0, p, 0.5f, 0);
        for (int r = -1; r <= 1; r += 2)
            AddPart(MESH_TORUS, MatTRS(V3(s*10.0f, 0, r*4.0f), V3(0, 0, 0), V3(2.5f, 2.5f, 2.5f)), dark, 0, PAT_PANEL, 0.4f, 0);
        ColCapsule(V3(s*10.0f, 0, -7.4f), V3(s*10.0f, 0, 7.4f), 2.4f);
        // porthole windows, some flickering
        for (int w = 0; w < 5; w++) {
            float z = -6 + w*3.0f;
            int blink = (w + (s > 0)) % 3 == 0 ? 2 : 0;
            Color lit = (w == 2 && s > 0) ? Col(40, 30, 30) : Col(255, 205, 120);
            AddPart(MESH_CUBE, MatTRS(V3(s*10.0f, 2.38f, z), V3(0, 0, 0), V3(0.55f, 0.08f, 0.38f)), lit, 1, PAT_NONE, 0, blink);
        }
    }
    // the breach: scorched hole with torn shards
    AddPart(MESH_SPHERE, MatTRS(V3(10, 2.25f, 4), V3(0.1f, 0, 0.2f), V3(1.5f, 0.45f, 1.7f)), Col(8, 6, 6), 0, PAT_NONE, 0, 0);
    for (int i = 0; i < 7; i++) {
        float a = i*0.9f;
        Vector3 c = V3(10 + cosf(a)*1.4f, 2.45f, 4 + sinf(a)*1.5f);
        AddPart(MESH_CUBE, MatTRS(c, V3(Rf(-0.8f, 0.8f), a, Rf(-0.9f, 0.9f)), V3(0.5f, 0.05f, 0.9f)), gold, 0, PAT_FOIL, 0.6f, 0);
    }

    // solar wings, one bent and broken
    for (int s = -1; s <= 1; s += 2) {
        AddSeg(V3(s*25.5f, 0, -5), V3(s*25.5f, 0, 5), 0.2f, dark, 0, PAT_NONE, 0.5f);
        for (int z = -1; z <= 1; z += 2) {
            Vector3 c = V3(s*31.0f, 0, z*2.7f);
            Vector3 e = V3(s*0.35f, 0, 0);
            if (s > 0 && z > 0) { c = V3(30.5f, -1.4f, 3.4f); e = V3(0.5f, 0.3f, -0.45f); }
            AddBox(c, e, V3(10, 0.12f, 3.8f), Col(40, 60, 140), 0, PAT_SOLAR, 0.9f, true);
            AddBox(c, e, V3(10.2f, 0.08f, 0.2f), dark, 0, PAT_NONE, 0.4f, false);
        }
    }

    // spinning-habitat ring with spokes
    AddPart(MESH_TORUS, MatTRS(V3(0, 0, 0), V3(0, PI/2, 0), V3(9, 9, 9)), hull, 0, PAT_PANEL, 0.3f, 0);
    for (int i = 0; i < 4; i++) {
        float a = i*PI/2 + PI/4;
        AddSeg(V3(0, cosf(a)*3.8f, sinf(a)*3.8f), V3(0, cosf(a)*8.6f, sinf(a)*8.6f), 0.35f, dark, 0, PAT_PANEL, 0.4f);
    }
    for (int i = 0; i < 20; i++) {
        float a = i*2*PI/20;
        ColSphere(V3(0, cosf(a)*9, sinf(a)*9), 1.3f);
    }

    // antenna mast and dish
    AddSeg(V3(0, 4, 0), V3(0, 8.6f, 0), 0.18f, dark, 0, PAT_NONE, 0.5f);
    AddPart(MESH_CONE, MatTRS(V3(0, 9.0f, 0), V3(PI, 0, 0.25f), V3(2.4f, 0.8f, 2.4f)), white, 0, PAT_PANEL, 0.4f, 0);
    ColCapsule(V3(0, 4, 0), V3(0, 8.8f, 0), 1.2f);
    AddPart(MESH_SPHERE_LO, MatTRS(ANTENNA_TIP, V3(0, 0, 0), V3(0.22f, 0.22f, 0.22f)), Col(255, 40, 30), 1, PAT_NONE, 0, 1);

    // docking tunnel
    AddSeg(V3(0, 0, -3.8f), V3(0, 0, -12), 1.3f, hull, 0, PAT_PANEL, 0.3f);
    AddPart(MESH_TORUS, MatTRS(V3(0, 0, -12), V3(0, 0, 0), V3(1.7f, 1.7f, 1.7f)), Col(255, 190, 60), 0.7f, PAT_NONE, 0.3f, 0);
    ColCapsule(V3(0, 0, -3.8f), V3(0, 0, -12), 1.6f);

    // red beacons on the truss ends
    for (int s = -1; s <= 1; s += 2)
        AddPart(MESH_SPHERE_LO, MatTRS(V3(s*26.4f, 0, 0), V3(0, 0, 0), V3(0.3f, 0.3f, 0.3f)), Col(255, 40, 30), 1, PAT_NONE, 0, 1);

    // ship TUG-7: hull, cockpit, engines and the glowing cargo bay
    Vector3 S = SHIP_POS;
    Color rust = Col(205, 108, 45);
    AddBox(S, V3(0, 0, 0), V3(5, 3, 8), rust, 0, PAT_PANEL, 0.35f, true);
    AddBox(Vector3Add(S, V3(0, 2.0f, -0.8f)), V3(0, 0, 0), V3(3.2f, 1.2f, 5), Col(120, 122, 128), 0, PAT_PANEL, 0.4f, true);
    AddPart(MESH_SPHERE, MatTRS(Vector3Add(S, V3(0, 1.0f, -3.9f)), V3(0, 0, 0), V3(1.9f, 1.3f, 1.6f)), Col(60, 210, 255), 0.25f, PAT_GLASS, 1, 0);
    for (int s = -1; s <= 1; s += 2) {
        Vector3 a = Vector3Add(S, V3(s*1.7f, -0.6f, -3.8f)), b = Vector3Add(S, V3(s*1.7f, -0.6f, -6.2f));
        AddSeg(a, b, 0.95f, Col(90, 92, 100), 0, PAT_PANEL, 0.5f);
        AddPart(MESH_SPHERE, MatTRS(b, V3(0, 0, 0), V3(0.8f, 0.8f, 0.25f)), Col(90, 160, 255), 1, PAT_NONE, 0, 0);
        AddSeg(Vector3Add(S, V3(s*2.6f, 1.2f, 3.8f)), Vector3Add(S, V3(s*2.6f, 1.2f, 4.3f)), 0.12f, s < 0 ? Col(255, 40, 40) : Col(40, 255, 80), 1, PAT_NONE, 0);
    }
    AddSeg(Vector3Add(S, V3(1.0f, 2.6f, -2)), Vector3Add(S, V3(1.3f, 5.0f, -2.5f)), 0.06f, Col(60, 60, 60), 0, PAT_NONE, 0.5f);
    // bay frame
    Vector3 B = Vector3Add(S, V3(0, 0, 4.2f));
    Color bayC = Col(60, 255, 130);
    AddSeg(Vector3Add(B, V3(-2.3f, -1.6f, 0)), Vector3Add(B, V3(2.3f, -1.6f, 0)), 0.14f, bayC, 1, PAT_NONE, 0);
    AddSeg(Vector3Add(B, V3(-2.3f, 1.6f, 0)), Vector3Add(B, V3(2.3f, 1.6f, 0)), 0.14f, bayC, 1, PAT_NONE, 0);
    AddSeg(Vector3Add(B, V3(-2.3f, -1.6f, 0)), Vector3Add(B, V3(-2.3f, 1.6f, 0)), 0.14f, bayC, 1, PAT_NONE, 0);
    AddSeg(Vector3Add(B, V3(2.3f, -1.6f, 0)), Vector3Add(B, V3(2.3f, 1.6f, 0)), 0.14f, bayC, 1, PAT_NONE, 0);
    ColCapsule(Vector3Add(S, V3(-1.7f, -0.6f, -3.8f)), Vector3Add(S, V3(-1.7f, -0.6f, -6.2f)), 1.0f);
    ColCapsule(Vector3Add(S, V3(1.7f, -0.6f, -3.8f)), Vector3Add(S, V3(1.7f, -0.6f, -6.2f)), 1.0f);
}

// ------------------------------------------------------------------ collision
static bool ClosestOnCollider(const Collider *k, Vector3 p, Vector3 *closest, Vector3 *normal, float *dist)
{
    switch (k->kind) {
    case COL_SPHERE: {
        Vector3 d = Vector3Subtract(p, k->a);
        float l = Vector3Length(d);
        *normal = l > 1e-5f ? Vector3Scale(d, 1/l) : V3(0, 1, 0);
        *dist = l - k->r;
        *closest = Vector3Add(k->a, Vector3Scale(*normal, k->r));
        return true;
    }
    case COL_CAPSULE: {
        Vector3 ab = Vector3Subtract(k->b, k->a);
        float t = Clamp(Vector3DotProduct(Vector3Subtract(p, k->a), ab)/Vector3DotProduct(ab, ab), 0, 1);
        Vector3 c = Vector3Add(k->a, Vector3Scale(ab, t));
        Vector3 d = Vector3Subtract(p, c);
        float l = Vector3Length(d);
        *normal = l > 1e-5f ? Vector3Scale(d, 1/l) : V3(0, 1, 0);
        *dist = l - k->r;
        *closest = Vector3Add(c, Vector3Scale(*normal, k->r));
        return true;
    }
    case COL_BOX: {
        Vector3 lp = Vector3Transform(p, k->inv);
        Vector3 c = V3(Clamp(lp.x, -k->half.x, k->half.x), Clamp(lp.y, -k->half.y, k->half.y), Clamp(lp.z, -k->half.z, k->half.z));
        Vector3 d = Vector3Subtract(lp, c);
        float l = Vector3Length(d);
        Vector3 nl;
        if (l > 1e-5f) { nl = Vector3Scale(d, 1/l); *dist = l; }
        else {
            // inside: push out through the nearest face
            float dx = k->half.x - fabsf(lp.x), dy = k->half.y - fabsf(lp.y), dz = k->half.z - fabsf(lp.z);
            if (dx < dy && dx < dz) nl = V3(lp.x < 0 ? -1.0f : 1.0f, 0, 0), *dist = -dx;
            else if (dy < dz) nl = V3(0, lp.y < 0 ? -1.0f : 1.0f, 0), *dist = -dy;
            else nl = V3(0, 0, lp.z < 0 ? -1.0f : 1.0f), *dist = -dz;
        }
        *normal = Vector3Transform(nl, k->rot);
        *closest = Vector3Subtract(p, Vector3Scale(*normal, *dist));
        return true;
    }
    }
    return false;
}

// Pushes a sphere out of the station; returns the impact speed (0 if no hit).
static float CollideSphere(Vector3 *pos, Vector3 *vel, float r, float bounce)
{
    float impact = 0;
    for (int i = 0; i < colCount; i++) {
        Vector3 c, n; float d;
        ClosestOnCollider(&cols[i], *pos, &c, &n, &d);
        if (d < r) {
            *pos = Vector3Add(*pos, Vector3Scale(n, r - d));
            float vn = Vector3DotProduct(*vel, n);
            if (vn < 0) {
                impact = fmaxf(impact, -vn);
                *vel = Vector3Subtract(*vel, Vector3Scale(n, (1 + bounce)*vn));
                *vel = Vector3Scale(*vel, 0.92f);
            }
        }
    }
    return impact;
}

static float RaySphere(Ray r, Vector3 c, float rad)
{
    Vector3 oc = Vector3Subtract(r.position, c);
    float b = Vector3DotProduct(oc, r.direction);
    float cc = Vector3DotProduct(oc, oc) - rad*rad;
    float h = b*b - cc;
    if (h < 0) return -1;
    float t = -b - sqrtf(h);
    if (t < 0) t = -b + sqrtf(h);
    return t;
}

static float RayCapsule(Ray r, Vector3 pa, Vector3 pb, float ra)
{
    Vector3 ba = Vector3Subtract(pb, pa), oa = Vector3Subtract(r.position, pa);
    float baba = Vector3DotProduct(ba, ba), bard = Vector3DotProduct(ba, r.direction), baoa = Vector3DotProduct(ba, oa);
    float rdoa = Vector3DotProduct(r.direction, oa), oaoa = Vector3DotProduct(oa, oa);
    float a = baba - bard*bard, b = baba*rdoa - baoa*bard, c = baba*oaoa - baoa*baoa - ra*ra*baba;
    float h = b*b - a*c;
    if (h >= 0) {
        float t = (-b - sqrtf(h))/a;
        float y = baoa + t*bard;
        if (y > 0 && y < baba && t > 0) return t;
        Vector3 oc = y <= 0 ? oa : Vector3Subtract(r.position, pb);
        b = Vector3DotProduct(r.direction, oc);
        c = Vector3DotProduct(oc, oc) - ra*ra;
        h = b*b - c;
        if (h > 0) { t = -b - sqrtf(h); if (t > 0) return t; }
    }
    return -1;
}

static float RayBox(Ray r, const Collider *k)
{
    Vector3 o = Vector3Transform(r.position, k->inv);
    Vector3 d = Vector3Subtract(Vector3Transform(Vector3Add(r.position, r.direction), k->inv), o);
    float tmin = -1e9f, tmax = 1e9f;
    float oo[3] = { o.x, o.y, o.z }, dd[3] = { d.x, d.y, d.z }, hh[3] = { k->half.x, k->half.y, k->half.z };
    for (int i = 0; i < 3; i++) {
        if (fabsf(dd[i]) < 1e-6f) { if (fabsf(oo[i]) > hh[i]) return -1; continue; }
        float t1 = (-hh[i] - oo[i])/dd[i], t2 = (hh[i] - oo[i])/dd[i];
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = fmaxf(tmin, t1); tmax = fminf(tmax, t2);
    }
    if (tmax < tmin || tmax < 0) return -1;
    return tmin > 0 ? tmin : tmax;
}

static Target CastRay(Ray r, float maxD)
{
    Target best = { TGT_NONE, -1, maxD, { 0 } };
    for (int i = 0; i < colCount; i++) {
        const Collider *k = &cols[i];
        float t = k->kind == COL_SPHERE ? RaySphere(r, k->a, k->r) : k->kind == COL_CAPSULE ? RayCapsule(r, k->a, k->b, k->r) : RayBox(r, k);
        if (t > 0 && t < best.dist) best = (Target){ TGT_STATION, i, t, { 0 } };
    }
    for (int i = 0; i < itemCount; i++) {
        if (!items[i].active) continue;
        float t = RaySphere(r, items[i].pos, items[i].radius + 0.35f);
        if (t > 0 && t < best.dist) best = (Target){ TGT_ITEM, i, t, { 0 } };
    }
    for (int i = 0; i < gloopCount; i++) {
        if (gloops[i].state == G_DEAD) continue;
        float t = RaySphere(r, gloops[i].pos, gloops[i].size + 0.3f);
        if (t > 0 && t < best.dist) best = (Target){ TGT_GLOOP, i, t, { 0 } };
    }
    best.point = Vector3Add(r.position, Vector3Scale(r.direction, best.dist));
    return best;
}

// ------------------------------------------------------------------ items
static const char *ItemName(const Item *it)
{
    switch (it->type) {
    case IT_SCRAP: return "Hull Scrap";
    case IT_BATTERY: return "Power Cell";
    case IT_CORE: return "Reactor Core";
    case IT_TOILET: return "Captain's Golden Throne";
    case IT_BOX: return it->awake ? "MIMIC!" : "Mystery Crate";
    case IT_O2: return "O2 Canister";
    case IT_TONGUE: return "Mimic Tongue (still wiggly)";
    }
    return "?";
}

static int AddItem(ItemType type, Vector3 pos, bool bolted)
{
    if (itemCount >= MAX_ITEMS) return -1;
    Item it = { .active = true, .type = type, .pos = pos, .bolted = bolted, .tether = -1 };
    it.rot = V3(Rf(0, 6), Rf(0, 6), Rf(0, 6));
    if (!bolted) {
        it.spin = V3(Rf(-0.6f, 0.6f), Rf(-0.6f, 0.6f), Rf(-0.6f, 0.6f));
        it.vel = V3(Rf(-0.15f, 0.15f), Rf(-0.15f, 0.15f), Rf(-0.15f, 0.15f));
    }
    switch (type) {
    case IT_SCRAP: it.radius = 0.75f; it.mass = 1; it.value = 20 + (rand() % 6)*5; break;
    case IT_BATTERY: it.radius = 0.55f; it.mass = 2; it.value = 60; break;
    case IT_CORE: it.radius = 0.7f; it.mass = 5; it.value = 200; break;
    case IT_TOILET: it.radius = 0.95f; it.mass = 6; it.value = 300; break;
    case IT_BOX: it.radius = 0.7f; it.mass = 3; it.value = 60 + (rand() % 9)*10; it.hp = 1; break;
    case IT_O2: it.radius = 0.5f; it.mass = 0.5f; it.value = 0; break;
    case IT_TONGUE: it.radius = 0.6f; it.mass = 2; it.value = 150; break;
    }
    items[itemCount] = it;
    return itemCount++;
}

static Vector3 FreeSpot(float rmin, float rmax)
{
    for (int tries = 0; tries < 50; tries++) {
        Vector3 d = Vector3Normalize(V3(Rf(-1, 1), Rf(-0.6f, 0.6f), Rf(-1, 1)));
        Vector3 p = Vector3Scale(d, Rf(rmin, rmax));
        bool ok = Vector3Distance(p, SHIP_POS) > 14;
        for (int i = 0; i < colCount && ok; i++) {
            Vector3 c, n; float dist;
            ClosestOnCollider(&cols[i], p, &c, &n, &dist);
            if (dist < 2.0f) ok = false;
        }
        if (ok) return p;
    }
    return V3(Rf(-20, 20), 14, Rf(-20, 20));
}

static void SpawnItems(void)
{
    itemCount = 0;
    AddItem(IT_CORE, V3(10, 3.2f, 4), true);
    AddItem(IT_TOILET, V3(-10, 0, 10.6f), true);
    AddItem(IT_BATTERY, V3(-18, 1.9f, 0), true);
    AddItem(IT_BATTERY, V3(18, -1.9f, 0), true);
    AddItem(IT_BATTERY, V3(0, 1.9f, -9), true);
    AddItem(IT_SCRAP, V3(0, -4.9f, 1.5f), true);
    AddItem(IT_SCRAP, V3(-10, -2.9f, -3), true);
    AddItem(IT_CORE, V3(-22, 12, 20), false);
    for (int i = 0; i < 8; i++) AddItem(IT_SCRAP, FreeSpot(12, 34), false);
    for (int i = 0; i < 2; i++) AddItem(IT_BATTERY, FreeSpot(14, 34), false);
    for (int i = 0; i < 5; i++) AddItem(IT_O2, FreeSpot(10, 30), false);
    int boxes = 5, mimics = 0;
    for (int i = 0; i < boxes; i++) {
        int k = AddItem(IT_BOX, FreeSpot(12, 32), false);
        bool m = Rf(0, 1) < dp.mimicChance || (i == boxes - 1 && mimics == 0);
        if (k >= 0 && m) { items[k].mimic = true; mimics++; }
    }
}

// ------------------------------------------------------------------ setup
static void ResetGlooper(Glooper *g, int vent)
{
    g->home = VENTS[vent % VENT_COUNT];
    g->pos = Vector3Add(g->home, V3(Rf(-1, 1), Rf(-1, 1), Rf(-1, 1)));
    g->vel = V3(0, 0, 0);
    g->state = G_LURK;
    g->hp = 1;
    g->timer = 0;
    g->giggleT = Rf(2, 6);
    static const Color palette[3] = { { 120, 235, 90, 255 }, { 190, 110, 245, 255 }, { 255, 115, 170, 255 } };
    g->col = palette[rand() % 3];
    g->size = Rf(0.9f, 1.3f);
    g->phase = Rf(0, 10);
}

void WorldNew(Difficulty d)
{
    diff = d;
    dp = DIFFS[d];
    srand(7 + d);
    BuildStation();
    SpawnItems();
    gloopCount = dp.gloopers;
    for (int i = 0; i < gloopCount; i++) ResetGlooper(&gloops[i], i);
    for (int i = 0; i < MAX_TETHERS; i++) tethers[i].active = false;
    ParticlesClear();

    pl = (Player){ 0 };
    pl.pos = Vector3Add(BAY_POS, V3(0, 0.5f, 5));
    pl.yaw = 0; pl.pitch = 0.05f;
    pl.bodyFwd = V3(0, 0, 1); pl.bodyUp = V3(0, 1, 0);
    pl.o2 = 100;

    stats = (RunStats){ .quota = dp.quota, .o2 = 100, .reserve = dp.reserve };
    worldTime = 0; deadT = 0; winT = 0; camShake = 0;
    dead = won = quotaAnnounced = false;
    saidLow = saidCrit = saidReserve = saidGloop = false;
    boltHintCd = farCd = bumpCd = 0; nearGloopCd = 0;
    camPosSmooth = Vector3Add(pl.pos, V3(0, 2, -6));
    char msg[200];
    snprintf(msg, sizeof msg, "Morning, meat-bag! Quota is $%d. Tether junk and haul it to the green bay. Try not to die, the paperwork is awful.", dp.quota);
    Radio(msg);
}

// ------------------------------------------------------------------ player helpers
static Vector3 ViewFwd(void)
{
    return V3(sinf(pl.yaw)*cosf(pl.pitch), sinf(pl.pitch), cosf(pl.yaw)*cosf(pl.pitch));
}

static Vector3 ViewRight(void) { return Vector3Normalize(Vector3CrossProduct(ViewFwd(), V3(0, 1, 0))); }

static Matrix BodyMatrix(void)
{
    Vector3 f = pl.bodyFwd;
    Vector3 u = Vector3Normalize(Vector3Subtract(pl.bodyUp, Vector3Scale(f, Vector3DotProduct(pl.bodyUp, f))));
    Vector3 r = Vector3CrossProduct(u, f);
    return MatBasis(r, u, f, pl.pos);
}

static Vector3 HandPos(void) { return Vector3Transform(V3(-0.42f, 0.0f, 0.55f), BodyMatrix()); }
static Vector3 PackPos(void) { return Vector3Transform(V3(0, 0.25f, -0.45f), BodyMatrix()); }

static int TetherCount(void)
{
    int n = 0;
    for (int i = 0; i < MAX_TETHERS; i++) n += tethers[i].active;
    return n;
}

static void ReleaseTether(int t)
{
    if (!tethers[t].active) return;
    if (tethers[t].item >= 0) items[tethers[t].item].tether = -1;
    tethers[t].active = false;
}

static void Burst(Vector3 p, int n, float speed, float life, float size, Color c)
{
    for (int i = 0; i < n; i++) {
        Vector3 v = Vector3Scale(Vector3Normalize(V3(Rf(-1, 1), Rf(-1, 1), Rf(-1, 1))), Rf(0.2f, 1.0f)*speed);
        ParticleSpawn(p, v, Rf(0.5f, 1.0f)*life, size*Rf(0.6f, 1.2f), c);
    }
}

static void WakeMimic(Item *it)
{
    if (it->awake) return;
    it->awake = true;
    it->hopT = 0.6f;
    if (it->tether >= 0) ReleaseTether(it->tether);
    SfxPlay(SFX_ROAR, 0.9f, 1.0f);
    static const char *lines[] = {
        "THAT CRATE HAS TEETH. Why does the crate have teeth?!",
        "Mimic! Torch it with LEFT MOUSE! Do NOT let it lick you!",
        "Oh no. Oh no no no. The box is hungry.",
    };
    RadioPick(lines, 3);
    camShake = fmaxf(camShake, 0.4f);
}

static void Hurt(float o2, Vector3 from, float knock, const char *cause)
{
    pl.o2 -= o2;
    pl.hurt = 1;
    Vector3 away = Vector3Normalize(Vector3Subtract(pl.pos, from));
    pl.vel = Vector3Add(pl.vel, Vector3Scale(away, knock));
    camShake = fmaxf(camShake, 0.6f);
    stats.deathCause = cause;
}

// ------------------------------------------------------------------ update
static void UpdatePlayer(float dt, const Input *in)
{
    float sens = settings.mouseSens*0.0022f;
    pl.yaw -= in->look.x*sens;
    pl.pitch -= in->look.y*sens*(settings.invertY ? -1 : 1);
    pl.pitch = Clamp(pl.pitch, -1.35f, 1.35f);

    Vector3 f = ViewFwd(), r = ViewRight(), u = Vector3CrossProduct(r, f);
    int latched = 0;
    for (int i = 0; i < gloopCount; i++) latched += gloops[i].state == G_LATCHED;

    Vector3 thrust = Vector3Add(Vector3Add(Vector3Scale(f, in->fwd), Vector3Scale(r, in->right)), Vector3Scale(u, in->up));
    float tl = Vector3Length(thrust);
    float accel = in->boost ? 11.0f : 6.0f;
    if (latched) accel *= 0.55f;
    if (tl > 0.01f) {
        thrust = Vector3Scale(thrust, 1.0f/fmaxf(tl, 1.0f));
        pl.vel = Vector3Add(pl.vel, Vector3Scale(thrust, accel*dt));
        pl.o2 -= (in->boost ? 1.3f : 0.35f)*dt;
        if (rand() % 2 == 0) {
            Vector3 back = PackPos();
            Vector3 pv = Vector3Add(pl.vel, Vector3Scale(thrust, -Rf(4, 7)));
            ParticleSpawn(back, Vector3Add(pv, V3(Rf(-0.5f, 0.5f), Rf(-0.5f, 0.5f), Rf(-0.5f, 0.5f))), 0.45f, 0.35f,
                          in->boost ? (Color){ 255, 170, 80, 200 } : (Color){ 170, 210, 255, 160 });
        }
    }
    if (in->brake) {
        pl.vel = Vector3Scale(pl.vel, expf(-2.5f*dt));
        pl.o2 -= 0.3f*dt;
    }
    float cap = in->boost ? 15.0f : 11.0f;
    float sp = Vector3Length(pl.vel);
    if (sp > cap) pl.vel = Vector3Scale(pl.vel, 1.0f - fminf(1.0f, (sp - cap)/sp*4*dt));

    // soft leash so the player can't drift away forever
    float distHome = Vector3Length(pl.pos);
    if (distHome > 110) {
        pl.vel = Vector3Add(pl.vel, Vector3Scale(Vector3Normalize(pl.pos), -(distHome - 110)*0.5f*dt));
        if (farCd <= 0) { Radio("You're drifting into the void. The void doesn't pay well. Turn around!"); farCd = 20; }
    }

    pl.pos = Vector3Add(pl.pos, Vector3Scale(pl.vel, dt));
    float impact = CollideSphere(&pl.pos, &pl.vel, PLAYER_R, 0.3f);
    if (impact > 2.5f && bumpCd <= 0) {
        SfxPlay(SFX_BUMP, fminf(1, impact/8), 1);
        camShake = fmaxf(camShake, fminf(0.5f, impact*0.05f));
        bumpCd = 0.3f;
        if (impact > 7) Radio("Careful! That station is older than me, and I'm held together with tape.");
    }

    // body follows the view, leaning into thrust
    pl.thrustVis += ((tl > 0.01f ? 1.0f : 0.0f) - pl.thrustVis)*fminf(1, dt*6);
    pl.lean += (in->fwd - pl.lean)*fminf(1, dt*4);
    pl.strafe += (in->right - pl.strafe)*fminf(1, dt*4);
    Vector3 wantF = Vector3Normalize(Vector3Add(f, Vector3Scale(u, -0.25f*pl.lean)));
    pl.bodyFwd = Vector3Normalize(Vector3Lerp(pl.bodyFwd, wantF, fminf(1, dt*7)));
    Vector3 wantU = Vector3Normalize(Vector3Add(u, Vector3Scale(r, 0.35f*pl.strafe)));
    pl.bodyUp = Vector3Normalize(Vector3Lerp(pl.bodyUp, wantU, fminf(1, dt*5)));

    // shaking off huggers
    if (latched && in->boostPressed) {
        pl.shakeCount++;
        camShake = fmaxf(camShake, 0.25f);
        if (pl.shakeCount >= 5) {
            for (int i = 0; i < gloopCount; i++) {
                Glooper *g = &gloops[i];
                if (g->state != G_LATCHED) continue;
                g->state = G_STUNNED;
                g->timer = 3.5f;
                g->vel = Vector3Add(pl.vel, Vector3Scale(Vector3Normalize(Vector3Subtract(g->pos, pl.pos)), 11));
            }
            pl.shakeCount = 0;
            SfxPlay(SFX_POP, 0.6f, 0.7f);
            Radio("You flung it off! It's spinning away, giggling. Why is it always giggling?");
        }
    }
    if (!latched) pl.shakeCount = 0;
}

static void UpdateTools(float dt, const Input *in)
{
    Ray ray = { cam.position, Vector3Normalize(Vector3Subtract(cam.target, cam.position)) };
    target = CastRay(ray, 60);
    float distFromPlayer = Vector3Distance(target.point, pl.pos);

    pl.cutting = in->cut;
    pl.cutHit = false;
    if (pl.cutting) {
        pl.o2 -= 0.25f*dt;
        Vector3 hand = HandPos();
        Vector3 aim = target.kind != TGT_NONE ? target.point : Vector3Add(ray.position, Vector3Scale(ray.direction, 60));
        Vector3 dir = Vector3Subtract(aim, hand);
        float len = Vector3Length(dir);
        dir = Vector3Scale(dir, 1/len);
        pl.cutEnd = Vector3Add(hand, Vector3Scale(dir, fminf(len, CUT_RANGE)));
        if (target.kind != TGT_NONE && distFromPlayer < CUT_RANGE) {
            pl.cutHit = true;
            if (rand() % 2 == 0)
                ParticleSpawn(target.point, V3(Rf(-3, 3), Rf(-3, 3), Rf(-3, 3)), 0.4f, 0.18f, (Color){ 255, 180, 70, 255 });
            if (target.kind == TGT_ITEM) {
                Item *it = &items[target.index];
                if (it->type == IT_BOX && it->mimic && !it->awake) WakeMimic(it);
                else if (it->type == IT_BOX && it->awake) {
                    it->hp -= dt/1.1f;
                    if (rand() % 3 == 0) ParticleSpawn(it->pos, V3(Rf(-2, 2), Rf(-2, 2), Rf(-2, 2)), 0.6f, 0.3f, (Color){ 255, 90, 160, 220 });
                    if (it->hp <= 0) {
                        Vector3 p = it->pos;
                        it->active = false;
                        Burst(p, 40, 5, 1.0f, 0.5f, (Color){ 255, 100, 170, 230 });
                        SfxPlay(SFX_POP, 1, 0.6f);
                        int k = AddItem(IT_TONGUE, p, false);
                        if (k >= 0) items[k].vel = V3(0, 0.5f, 0);
                        stats.mimicsSlain++;
                        Radio("You killed the crate. The crate is now a tongue. Tongues sell for $150. Nature is beautiful.");
                    }
                } else if (it->bolted) {
                    it->cut += dt/1.3f;
                    if (it->cut >= 1) {
                        it->bolted = false;
                        it->vel = Vector3Scale(Vector3Normalize(Vector3Subtract(it->pos, V3(0, 0, 0))), 1.2f);
                        it->spin = V3(Rf(-1, 1), Rf(-1, 1), Rf(-1, 1));
                        SfxPlay(SFX_UNBOLT, 0.9f, 1);
                        Burst(it->pos, 25, 4, 0.6f, 0.2f, (Color){ 255, 200, 90, 255 });
                    }
                }
            } else if (target.kind == TGT_GLOOP) {
                Glooper *g = &gloops[target.index];
                g->hp -= dt/0.55f;
                g->vel = Vector3Add(g->vel, Vector3Scale(ray.direction, 6*dt));
                if (g->state == G_LURK) g->state = G_CHASE;
                if (g->hp <= 0) {
                    Burst(g->pos, 45, 6, 1.1f, 0.55f, g->col);
                    SfxPlay(SFX_POP, 1, Rf(0.9f, 1.3f));
                    g->state = G_DEAD;
                    g->timer = 22;
                    stats.gloopsPopped++;
                    if (rand() % 2 == 0) {
                        static const char *lines[] = {
                            "Splat! Gross. Do it again.",
                            "Glooper down. It'll be back. They always come back.",
                            "You popped it! I'll add 'goo' to the cleaning list.",
                        };
                        RadioPick(lines, 3);
                    }
                }
            }
        }
    }

    // tethers
    if (in->tether && target.kind == TGT_ITEM && distFromPlayer < TETHER_RANGE) {
        Item *it = &items[target.index];
        if (it->type == IT_O2) {
            // canisters are grabbed by touching them
        } else if (it->bolted) {
            if (boltHintCd <= 0) { Radio("That's bolted down. Hold LEFT MOUSE to torch it loose. Safety third!"); boltHintCd = 8; }
        } else if (it->type == IT_BOX && it->awake) {
            Radio("You want to put a leash on THAT? Torch it first!");
        } else if (it->tether < 0) {
            int slot = -1;
            for (int i = 0; i < MAX_TETHERS; i++) if (!tethers[i].active) { slot = i; break; }
            if (slot < 0) {
                if (boltHintCd <= 0) { Radio("Three tethers max! Haul these home or let one go with E."); boltHintCd = 6; }
            } else {
                tethers[slot] = (Tether){ true, target.index, Clamp(Vector3Distance(it->pos, pl.pos), 3, 9) };
                it->tether = slot;
                SfxPlay(SFX_TETHER, 0.8f, 1);
                if (it->type == IT_BOX && it->mimic) WakeMimic(it);
            }
        }
    }
    if (in->release && TetherCount() > 0) {
        for (int i = 0; i < MAX_TETHERS; i++) ReleaseTether(i);
        SfxPlay(SFX_RELEASE, 0.7f, 1);
    }

    for (int t = 0; t < MAX_TETHERS; t++) {
        Tether *te = &tethers[t];
        if (!te->active) continue;
        Item *it = &items[te->item];
        if (!it->active) { te->active = false; continue; }
        if (in->reel) te->rest = fmaxf(2.5f, te->rest - 5*dt);
        Vector3 d = Vector3Subtract(it->pos, pl.pos);
        float dist = Vector3Length(d);
        if (dist > te->rest && dist > 1e-4f) {
            Vector3 n = Vector3Scale(d, 1/dist);
            float rel = Vector3DotProduct(Vector3Subtract(it->vel, pl.vel), n);
            float force = fminf(60, 14*(dist - te->rest) + 3.5f*rel);
            if (force > 0) {
                pl.vel = Vector3Add(pl.vel, Vector3Scale(n, force/3.0f*dt));
                it->vel = Vector3Subtract(it->vel, Vector3Scale(n, force/it->mass*dt));
            }
        }
        if (dist > 40) { ReleaseTether(t); Radio("Tether snapped! Too far, too heavy, too much."); }
    }
}

static void UpdateItems(float dt)
{
    for (int i = 0; i < itemCount; i++) {
        Item *it = &items[i];
        if (!it->active) continue;
        it->rot = Vector3Add(it->rot, Vector3Scale(it->spin, dt));
        if (it->bolted) continue;

        if (it->type == IT_BOX && it->mimic && it->awake) {
            Vector3 to = Vector3Subtract(pl.pos, it->pos);
            float d = Vector3Length(to);
            it->hopT -= dt;
            it->chomp += dt*9;
            if (it->hopT <= 0 && !dead) {
                it->vel = Vector3Add(Vector3Scale(it->vel, 0.3f), Vector3Scale(to, 7.5f/fmaxf(d, 0.1f)));
                it->hopT = 0.8f;
                SfxPlay(SFX_BITE, 0.3f, 1.4f);
            }
            it->vel = Vector3Scale(it->vel, expf(-0.8f*dt));
            it->biteCd -= dt;
            if (d < it->radius + PLAYER_R + 0.2f && it->biteCd <= 0 && !dead) {
                Hurt(10, it->pos, 9, "chewed on by a crate");
                SfxPlay(SFX_BITE, 1, 1);
                it->biteCd = 1.8f;
                Burst(pl.pos, 15, 3, 0.6f, 0.3f, (Color){ 200, 230, 255, 200 });
            }
        } else if (it->type == IT_BOX && it->mimic && Vector3Distance(it->pos, pl.pos) < 5.5f) {
            WakeMimic(it);
        }

        it->pos = Vector3Add(it->pos, Vector3Scale(it->vel, dt));
        it->vel = Vector3Scale(it->vel, 1.0f - 0.05f*dt);
        CollideSphere(&it->pos, &it->vel, it->radius, 0.35f);
        float r = Vector3Length(it->pos);
        if (r > 100) it->vel = Vector3Add(it->vel, Vector3Scale(it->pos, -0.02f*dt));

        // oxygen canisters are grabbed on touch
        if (it->type == IT_O2 && Vector3Distance(it->pos, pl.pos) < it->radius + PLAYER_R + 0.4f) {
            it->active = false;
            pl.o2 = fminf(100, pl.o2 + 30);
            SfxPlay(SFX_O2, 0.8f, 1);
            Burst(it->pos, 20, 3, 0.7f, 0.3f, (Color){ 120, 220, 255, 220 });
            Radio("Fresh air! Breathe it in. Not too fast, you'll get the hiccups.");
            continue;
        }
        // deposit anything that floats into the bay
        if (it->type != IT_O2 && !(it->type == IT_BOX && it->awake) && Vector3Distance(it->pos, BAY_POS) < BAY_R) {
            if (it->tether >= 0) ReleaseTether(it->tether);
            it->active = false;
            stats.credits += it->value;
            stats.deposited++;
            SfxPlay(SFX_DEPOSIT, 0.9f, 1);
            Burst(BAY_POS, 30, 4, 0.8f, 0.4f, (Color){ 80, 255, 140, 230 });
            char msg[160];
            static const char *fmt[] = {
                "Ka-ching! $%d. I'm buying a new antenna.",
                "Ooh, $%d. That's going on the fridge.",
                "$%d! You're my favourite employee. You're my only employee.",
                "Another $%d. Mom would be proud. I don't have a mom. I have a toaster.",
            };
            if (it->type == IT_TOILET) snprintf(msg, sizeof msg, "The Golden Throne! $%d! I will sit on it. I have no legs. I will find a way.", it->value);
            else snprintf(msg, sizeof msg, fmt[rand() % 4], it->value);
            Radio(msg);
        }
    }
}

static void UpdateGloopers(float dt)
{
    float tension = 0;
    for (int i = 0; i < gloopCount; i++) {
        Glooper *g = &gloops[i];
        g->phase += dt;
        Vector3 to = Vector3Subtract(pl.pos, g->pos);
        float d = Vector3Length(to);
        switch (g->state) {
        case G_DEAD:
            g->timer -= dt;
            if (g->timer <= 0) ResetGlooper(g, rand());
            continue;
        case G_LURK: {
            Vector3 wander = V3(sinf(g->phase*0.7f + i), cosf(g->phase*0.5f + i*2), sinf(g->phase*0.6f + i*3));
            Vector3 home = Vector3Subtract(g->home, g->pos);
            g->vel = Vector3Add(g->vel, Vector3Scale(Vector3Add(wander, Vector3Scale(home, 0.3f)), dt));
            g->vel = Vector3Scale(g->vel, expf(-1.0f*dt));
            if (d < dp.gloopSense && !dead) {
                g->state = G_CHASE;
                if (!saidGloop) {
                    Radio("Uh. Something on the scanner is... giggling? Those are Gloopers. They love hugs. They hug the air right out of you.");
                    saidGloop = true;
                }
            }
            break;
        }
        case G_CHASE: {
            Vector3 dir = Vector3Scale(to, 1/fmaxf(d, 0.01f));
            Vector3 side = Vector3Normalize(Vector3CrossProduct(dir, V3(0, 1, 0)));
            Vector3 wob = Vector3Scale(side, sinf(g->phase*2.3f)*2.0f);
            g->vel = Vector3Add(g->vel, Vector3Scale(Vector3Add(Vector3Scale(dir, 3.5f), wob), dt));
            float sp = Vector3Length(g->vel);
            if (sp > dp.gloopSpeed) g->vel = Vector3Scale(g->vel, dp.gloopSpeed/sp);
            g->giggleT -= dt;
            if (g->giggleT <= 0) {
                SfxPlay(SFX_GIGGLE, Clamp(1.2f - d/30, 0.15f, 1), Rf(0.85f, 1.35f));
                g->giggleT = Rf(3, 7);
            }
            if (d > dp.gloopSense*1.8f) g->state = G_LURK;
            if (d < g->size + PLAYER_R + 0.1f && !dead) {
                g->state = G_LATCHED;
                Vector3 local = Vector3Normalize(V3(Rf(-0.8f, 0.8f), Rf(-0.2f, 0.9f), -1));
                g->latchOff = Vector3Scale(local, PLAYER_R + g->size*0.55f);
                SfxPlay(SFX_SLURP, 1, Rf(0.9f, 1.2f));
                pl.hurt = 0.8f;
                stats.deathCause = "hugged by a Glooper";
                static const char *lines[] = {
                    "IT'S HUGGING YOU! Mash SHIFT to wiggle free!",
                    "Glooper on your back! Shake it! SHAKE IT! (SHIFT!)",
                    "It's sucking your air like a milkshake! Mash SHIFT or torch it!",
                };
                RadioPick(lines, 3);
            }
            tension = fmaxf(tension, Clamp(1 - d/dp.gloopSense, 0, 1));
            break;
        }
        case G_LATCHED: {
            Matrix B = BodyMatrix();
            g->pos = Vector3Transform(g->latchOff, B);
            g->vel = pl.vel;
            pl.o2 -= dp.gloopDrain*dt;
            tension = 1;
            if (rand() % 6 == 0) {
                Vector3 p = Vector3Lerp(pl.pos, g->pos, 0.5f);
                ParticleSpawn(p, Vector3Scale(Vector3Subtract(g->pos, pl.pos), 1.5f), 0.5f, 0.2f, (Color){ 180, 230, 255, 180 });
            }
            continue;
        }
        case G_STUNNED:
            g->timer -= dt;
            g->vel = Vector3Scale(g->vel, expf(-0.7f*dt));
            if (g->timer <= 0) g->state = G_CHASE;
            break;
        }
        g->pos = Vector3Add(g->pos, Vector3Scale(g->vel, dt));
        CollideSphere(&g->pos, &g->vel, g->size, 0.5f);
    }
    float danger = Clamp((35 - pl.o2)/35, 0, 1);
    AudioSetLoops(pl.thrustVis, pl.cutting ? 1 : 0, dead ? 0 : danger, tension);
}

static void UpdateChaseCamera(float dt)
{
    Vector3 f = ViewFwd(), r = ViewRight(), u = Vector3CrossProduct(r, f);
    Vector3 pivot = Vector3Add(pl.pos, Vector3Scale(u, 0.9f));
    Vector3 want = Vector3Add(Vector3Add(pivot, Vector3Scale(f, -6.2f)), Vector3Add(Vector3Scale(u, 0.6f), Vector3Scale(r, 1.25f)));
    camPosSmooth = Vector3Lerp(camPosSmooth, want, fminf(1, dt*14));
    Vector3 p = camPosSmooth;
    if (camShake > 0 && settings.screenShake) {
        p = Vector3Add(p, V3(Rf(-1, 1)*camShake*0.3f, Rf(-1, 1)*camShake*0.3f, Rf(-1, 1)*camShake*0.3f));
    }
    cam.position = p;
    cam.target = Vector3Add(p, f);
    cam.up = u;
    cam.fovy = 68;
    cam.projection = CAMERA_PERSPECTIVE;
}

RunState WorldUpdate(float dt, const Input *in)
{
    if (dt > 0.05f) dt = 0.05f;
    worldTime += dt;
    stats.time += dt;
    boltHintCd -= dt; farCd -= dt; bumpCd -= dt; nearGloopCd -= dt;
    radioT += dt;
    pl.hurt = fmaxf(0, pl.hurt - dt*1.5f);
    camShake = fmaxf(0, camShake - dt*1.8f);

    Input none = { 0 };
    const Input *use = (dead || won) ? &none : in;
    if (dead) {
        pl.yaw += dt*0.6f;
        pl.pitch = sinf(worldTime)*0.6f;
    }
    UpdatePlayer(dt, use);
    UpdateChaseCamera(dt);
    UpdateTools(dt, use);
    UpdateItems(dt);
    UpdateGloopers(dt);
    ParticlesUpdate(dt);

    if (!dead && !won) {
        pl.o2 -= dp.o2Drain*dt;
        bool inBay = Vector3Distance(pl.pos, BAY_POS) < BAY_R + 0.5f;
        if (inBay && pl.o2 < 100 && stats.reserve > 0) {
            float take = fminf(fminf(30*dt, 100 - pl.o2), stats.reserve);
            pl.o2 += take;
            stats.reserve -= take;
            if (rand() % 3 == 0) ParticleSpawn(Vector3Add(pl.pos, V3(Rf(-1, 1), Rf(-1, 1), Rf(-1, 1))), V3(0, 0, 0), 0.5f, 0.25f, (Color){ 100, 220, 255, 160 });
            if (stats.reserve <= 0 && !saidReserve) {
                Radio("Ship reserve is bone dry! Grab the blue O2 canisters floating around the station!");
                saidReserve = true;
            }
        }
        if (stats.credits >= stats.quota && !quotaAnnounced) {
            quotaAnnounced = true;
            SfxPlay(SFX_ALARM, 0.6f, 1.3f);
            Radio("QUOTA MET! Get your butt back to the bay and we can go home!");
        }
        if (inBay && stats.credits >= stats.quota) {
            won = true;
            winT = 0;
            for (int i = 0; i < MAX_TETHERS; i++) ReleaseTether(i);
            SfxPlay(SFX_WIN, 1, 1);
            stats.score = stats.credits + (int)(pl.o2*5) + (int)(stats.reserve*2);
            Radio("Shift complete! Strap in. I'll drive. I've only crashed twice.");
        }
        if (pl.o2 < 30 && !saidLow) { Radio("Your air is getting thin. Head back to the green bay to refill!"); saidLow = true; SfxPlay(SFX_ALARM, 0.7f, 1); }
        if (pl.o2 > 45) saidLow = false;
        if (pl.o2 < 15 && !saidCrit) { Radio("OXYGEN CRITICAL. I'm legally required to say: please don't die."); saidCrit = true; SfxPlay(SFX_ALARM, 0.9f, 1.2f); }
        if (pl.o2 > 25) saidCrit = false;
        if (pl.o2 <= 0) {
            pl.o2 = 0;
            dead = true;
            deadT = 0;
            if (!stats.deathCause) stats.deathCause = "ran out of air";
            for (int i = 0; i < MAX_TETHERS; i++) ReleaseTether(i);
            SfxPlay(SFX_DEATH, 1, 1);
            Radio("...meat-bag? Hello? Oh no. I'll get the space mop.");
        }
    }
    stats.o2 = pl.o2;
    if (dead) { deadT += dt; if (deadT > 3.2f) return RUN_DEAD; }
    if (won) { winT += dt; if (winT > 2.2f) return RUN_WON; }
    return RUN_PLAYING;
}

void WorldIdle(float dt)
{
    worldTime += dt;
    for (int i = 0; i < itemCount; i++)
        if (items[i].active && !items[i].bolted) items[i].rot = Vector3Add(items[i].rot, Vector3Scale(items[i].spin, dt));
    for (int i = 0; i < gloopCount; i++) {
        Glooper *g = &gloops[i];
        g->phase += dt;
        Vector3 wander = V3(sinf(g->phase*0.7f + i), cosf(g->phase*0.5f + i*2), sinf(g->phase*0.6f + i*3));
        g->pos = Vector3Add(g->home, Vector3Scale(wander, 2.5f));
    }
    ParticlesUpdate(dt);
    AudioSetLoops(0, 0, 0, 0.2f);
}

Camera3D WorldCamera(void) { return cam; }

Camera3D WorldMenuCamera(float t)
{
    Camera3D c = { 0 };
    float a = t*0.05f + 2.2f;
    c.position = V3(sinf(a)*48, 9 + sinf(t*0.11f)*5, cosf(a)*48);
    c.target = V3(0, -2, 0);
    c.up = V3(0, 1, 0);
    c.fovy = 55;
    c.projection = CAMERA_PERSPECTIVE;
    return c;
}

// ------------------------------------------------------------------ drawing
static void DrawEye(Matrix B, Vector3 c, float r, Vector3 lookLocal, float t, float seed)
{
    float blink = fmodf(t + seed*3.1f, 4.0f) < 0.12f ? 0.15f : 1.0f;
    GfxDraw(MESH_SPHERE, MatLocal(B, c, V3(0, 0, 0), V3(r, r*blink, r)), (Color){ 245, 245, 238, 255 }, 0.05f, PAT_NONE, 0.95f);
    Vector3 jig = V3(sinf(t*13 + seed)*0.25f, cosf(t*11 + seed*2)*0.22f, 0);
    Vector3 ld = Vector3Normalize(Vector3Add(lookLocal, jig));
    Vector3 pc = Vector3Add(c, Vector3Scale(ld, r*0.72f));
    if (blink > 0.5f) GfxDraw(MESH_SPHERE_LO, MatLocal(B, pc, V3(0, 0, 0), V3(r*0.45f, r*0.45f, r*0.45f)), (Color){ 8, 8, 10, 255 }, 0, PAT_NONE, 1);
}

static Matrix FacingBasis(Vector3 pos, Vector3 toward)
{
    Vector3 f = Vector3Normalize(Vector3Subtract(toward, pos));
    if (Vector3Length(f) < 0.5f) f = V3(0, 0, 1);
    Vector3 ref = fabsf(f.y) > 0.95f ? V3(1, 0, 0) : V3(0, 1, 0);
    Vector3 r = Vector3Normalize(Vector3CrossProduct(ref, f));
    Vector3 u = Vector3CrossProduct(f, r);
    return MatBasis(r, u, f, pos);
}

static void DrawGlooper(const Glooper *g, Vector3 lookAt)
{
    float t = g->phase;
    float s = g->size;
    Matrix B = FacingBasis(g->pos, lookAt);
    float w = sinf(t*5.0f);
    float hug = g->state == G_LATCHED ? 1.0f : 0.0f;
    Vector3 body = V3(s*(1 + 0.1f*w + 0.15f*hug), s*(1 - 0.1f*w), s*(1 + 0.08f*w - 0.2f*hug));
    Color flash = g->hp < 1 ? (Color){ 255, 255, 255, 255 } : g->col;
    Color c = g->state == G_STUNNED ? (Color){ (unsigned char)(g->col.r*0.6f), (unsigned char)(g->col.g*0.6f), (unsigned char)(g->col.b*0.6f), 255 } : g->col;
    if (g->hp < 0.999f && fmodf(t*20, 2) < 1) c = flash;
    GfxDraw(MESH_SPHERE, MatLocal(B, V3(0, 0, 0), V3(0, 0, 0), body), c, 0.05f, PAT_GOO, 0.8f);
    // big mismatched googly eyes
    Vector3 look = V3(0, 0, 1);
    DrawEye(B, V3(0.42f*s, 0.45f*s, 0.72f*s), 0.42f*s, look, t, 1);
    DrawEye(B, V3(-0.38f*s, 0.55f*s, 0.76f*s), 0.3f*s, look, t, 2);
    // grin with crooked teeth
    float open = g->state == G_CHASE ? 0.3f + 0.15f*sinf(t*7) : 0.22f;
    GfxDraw(MESH_SPHERE, MatLocal(B, V3(0, -0.22f*s, 0.84f*s), V3(0.15f, 0, 0), V3(0.62f*s, open*s, 0.24f*s)), (Color){ 45, 6, 18, 255 }, 0, PAT_NONE, 0.3f);
    for (int k = 0; k < 6; k++) {
        float x = (-0.45f + k*0.18f)*s;
        float len = (0.12f + 0.06f*((k*7) % 3))*s;
        GfxDraw(MESH_CONE, MatLocal(B, V3(x, (-0.22f + open*0.7f)*s, 0.98f*s), V3(PI, 0, 0), V3(0.06f*s, len, 0.06f*s)), (Color){ 250, 248, 230, 255 }, 0, PAT_NONE, 0.6f);
    }
    for (int k = 0; k < 4; k++) {
        float x = (-0.3f + k*0.2f)*s;
        GfxDraw(MESH_CONE, MatLocal(B, V3(x, (-0.22f - open*0.75f)*s, 0.96f*s), V3(0, 0, 0), V3(0.05f*s, 0.1f*s, 0.05f*s)), (Color){ 250, 248, 230, 255 }, 0, PAT_NONE, 0.6f);
    }
    // tongue lolling out of the side
    GfxDraw(MESH_SPHERE_LO, MatLocal(B, V3(0.22f*s, (-0.36f - 0.05f*sinf(t*3))*s, 1.0f*s), V3(0.4f + 0.2f*sinf(t*4), 0, 0.3f), V3(0.16f*s, 0.06f*s, 0.3f*s)), (Color){ 255, 110, 150, 255 }, 0, PAT_GOO, 0.9f);
    // wiggly tentacles
    for (int a = 0; a < 5; a++) {
        float ang = a*2*PI/5 + 0.3f;
        Vector3 base = V3(cosf(ang)*0.45f*s, -0.75f*s, sinf(ang)*0.45f*s - 0.15f*s);
        for (int j = 1; j <= 6; j++) {
            float sw = sinf(t*4 + j*0.7f + a*1.3f)*0.13f*j*s;
            Vector3 lp = V3(base.x*(1 + j*0.15f) + sw, base.y - j*0.24f*s, base.z*(1 + j*0.15f) - j*0.1f*s + cosf(t*3 + j + a)*0.08f*j*s);
            float rr = 0.2f*s*(1 - j/7.5f);
            GfxDraw(MESH_SPHERE_LO, MatLocal(B, lp, V3(0, 0, 0), V3(rr, rr, rr)), c, 0.03f, PAT_GOO, 0.8f);
        }
    }
}

static void DrawMimic(const Item *it, Vector3 lookAt)
{
    Matrix B = it->awake ? FacingBasis(it->pos, lookAt) : MatTRS(it->pos, it->rot, V3(1, 1, 1));
    Color wood = { 150, 96, 46, 255 }, band = { 70, 70, 76, 255 };
    if (!it->awake) {
        GfxDraw(MESH_CUBE, MatLocal(B, V3(0, 0, 0), V3(0, 0, 0), V3(1.1f, 1.1f, 1.1f)), wood, 0, PAT_PANEL, 0.2f);
        GfxDraw(MESH_CUBE, MatLocal(B, V3(0, 0, 0), V3(0, 0, 0), V3(1.14f, 0.16f, 1.14f)), band, 0, PAT_NONE, 0.6f);
        GfxDraw(MESH_CUBE, MatLocal(B, V3(0, 0, 0), V3(0, 0, 0), V3(0.16f, 1.14f, 1.14f)), band, 0, PAT_NONE, 0.6f);
        return;
    }
    float open = 0.45f + 0.35f*sinf(it->chomp);
    float hp = it->hp < 0.999f && fmodf(it->chomp*3, 2) < 1 ? 1.0f : 0.0f;
    Color w2 = hp > 0 ? (Color){ 255, 255, 255, 255 } : wood;
    // lower jaw
    GfxDraw(MESH_CUBE, MatLocal(B, V3(0, -0.28f, 0), V3(0, 0, 0), V3(1.1f, 0.55f, 1.1f)), w2, 0, PAT_PANEL, 0.2f);
    GfxDraw(MESH_CUBE, MatLocal(B, V3(0, -0.01f, 0), V3(0, 0, 0), V3(0.98f, 0.04f, 0.98f)), (Color){ 90, 10, 25, 255 }, 0, PAT_GOO, 0.6f);
    for (int k = 0; k < 6; k++)
        GfxDraw(MESH_CONE, MatLocal(B, V3(-0.42f + k*0.17f, 0, 0.47f), V3(0, 0, 0), V3(0.06f, 0.16f, 0.06f)), (Color){ 250, 248, 230, 255 }, 0, PAT_NONE, 0.6f);
    // tongue
    GfxDraw(MESH_SPHERE, MatLocal(B, V3(0, 0.02f, 0.65f), V3(-0.2f + 0.3f*sinf(it->chomp*0.7f), 0.3f*sinf(it->chomp*0.5f), 0), V3(0.26f, 0.08f, 0.6f)), (Color){ 255, 105, 150, 255 }, 0, PAT_GOO, 0.9f);
    // lid hinged at the back
    Matrix lid = MatrixMultiply(MatrixMultiply(MatrixTranslate(0, 0, 0.55f), MatrixRotateX(-open)), MatrixTranslate(0, 0, -0.55f));
    Matrix L = MatrixMultiply(lid, B);
    GfxDraw(MESH_CUBE, MatLocal(L, V3(0, 0.08f, 0), V3(0, 0, 0), V3(1.12f, 0.18f, 1.12f)), w2, 0, PAT_PANEL, 0.2f);
    GfxDraw(MESH_CUBE, MatLocal(L, V3(0, 0.08f, 0), V3(0, 0, 0), V3(1.15f, 0.2f, 0.16f)), band, 0, PAT_NONE, 0.6f);
    for (int k = 0; k < 6; k++)
        GfxDraw(MESH_CONE, MatLocal(L, V3(-0.42f + k*0.17f, -0.02f, 0.47f), V3(PI, 0, 0), V3(0.06f, 0.18f, 0.06f)), (Color){ 250, 248, 230, 255 }, 0, PAT_NONE, 0.6f);
    // eyes on wobbly stalks
    for (int s = -1; s <= 1; s += 2) {
        Vector3 root = V3(s*0.3f, 0.15f, -0.15f);
        Vector3 tip = V3(s*(0.5f + 0.08f*sinf(it->chomp*0.6f + s)), 0.85f, 0.1f);
        GfxSeg(Vector3Transform(root, L), Vector3Transform(tip, L), 0.05f, (Color){ 255, 120, 160, 255 }, 0, PAT_GOO, 0.8f);
        DrawEye(L, tip, s < 0 ? 0.24f : 0.19f, V3(0, -0.3f, 1), it->chomp*0.4f, (float)s + 3);
    }
}

static void DrawItem(const Item *it)
{
    if (it->type == IT_BOX && it->mimic) { DrawMimic(it, pl.pos); return; }
    Matrix B = MatTRS(it->pos, it->rot, V3(1, 1, 1));
    float t = worldTime;
    switch (it->type) {
    case IT_SCRAP:
        GfxDraw(MESH_CUBE, MatLocal(B, V3(0, 0, 0), V3(0, 0, 0), V3(1.3f, 0.1f, 0.9f)), (Color){ 160, 162, 168, 255 }, 0, PAT_PANEL, 0.4f);
        GfxDraw(MESH_CUBE, MatLocal(B, V3(0.55f, 0.25f, 0), V3(0, 0, 0.9f), V3(0.6f, 0.1f, 0.9f)), (Color){ 140, 142, 150, 255 }, 0, PAT_PANEL, 0.4f);
        GfxSeg(Vector3Transform(V3(-0.5f, 0.05f, 0.3f), B), Vector3Transform(V3(-0.9f, 0.5f, 0.6f), B), 0.04f, (Color){ 230, 100, 30, 255 }, 0, PAT_NONE, 0.5f);
        break;
    case IT_BATTERY:
        GfxSeg(Vector3Transform(V3(0, -0.45f, 0), B), Vector3Transform(V3(0, 0.45f, 0), B), 0.3f, (Color){ 230, 200, 40, 255 }, 0, PAT_NONE, 0.6f);
        GfxSeg(Vector3Transform(V3(0, -0.08f, 0), B), Vector3Transform(V3(0, 0.08f, 0), B), 0.32f, (Color){ 60, 255, 120, 255 }, 0.8f + 0.2f*sinf(t*4), PAT_NONE, 0);
        GfxSeg(Vector3Transform(V3(0, 0.45f, 0), B), Vector3Transform(V3(0, 0.6f, 0), B), 0.1f, (Color){ 90, 90, 95, 255 }, 0, PAT_NONE, 0.8f);
        break;
    case IT_CORE: {
        float pulse = 0.75f + 0.25f*sinf(t*3);
        GfxBall(it->pos, 0.42f, (Color){ 90, 230, 255, 255 }, pulse, PAT_NONE, 0);
        for (int k = 0; k < 3; k++)
            GfxDraw(MESH_TORUS, MatLocal(B, V3(0, 0, 0), V3(k*1.05f + t*0.8f, k*0.7f, t*0.5f*k), V3(0.62f, 0.62f, 0.62f)), (Color){ 120, 122, 130, 255 }, 0, PAT_NONE, 0.8f);
        break;
    }
    case IT_TOILET: {
        Color g = { 240, 190, 60, 255 };
        GfxDraw(MESH_SPHERE, MatLocal(B, V3(0, -0.1f, 0.1f), V3(0, 0, 0), V3(0.5f, 0.38f, 0.6f)), g, 0.05f, PAT_NONE, 0.95f);
        GfxDraw(MESH_TORUS, MatLocal(B, V3(0, 0.22f, 0.12f), V3(PI/2, 0, 0), V3(0.48f, 0.56f, 0.5f)), g, 0.05f, PAT_NONE, 0.95f);
        GfxDraw(MESH_CUBE, MatLocal(B, V3(0, 0.45f, -0.5f), V3(0, 0, 0), V3(0.85f, 0.75f, 0.32f)), g, 0.05f, PAT_NONE, 0.95f);
        GfxDraw(MESH_CUBE, MatLocal(B, V3(0, 0.62f, -0.12f), V3(-1.0f, 0, 0), V3(0.8f, 0.05f, 0.7f)), g, 0.05f, PAT_NONE, 0.95f);
        GfxSeg(Vector3Transform(V3(0, -0.75f, 0.05f), B), Vector3Transform(V3(0, -0.3f, 0.05f), B), 0.24f, g, 0.05f, PAT_NONE, 0.95f);
        break;
    }
    case IT_BOX:
        DrawMimic(it, pl.pos);
        break;
    case IT_O2:
        GfxSeg(Vector3Transform(V3(0, -0.45f, 0), B), Vector3Transform(V3(0, 0.4f, 0), B), 0.26f, (Color){ 235, 240, 245, 255 }, 0, PAT_NONE, 0.7f);
        GfxSeg(Vector3Transform(V3(0, 0.0f, 0), B), Vector3Transform(V3(0, 0.15f, 0), B), 0.28f, (Color){ 60, 190, 255, 255 }, 0.9f, PAT_NONE, 0);
        GfxSeg(Vector3Transform(V3(0, 0.4f, 0), B), Vector3Transform(V3(0, 0.58f, 0), B), 0.09f, (Color){ 80, 80, 85, 255 }, 0, PAT_NONE, 0.8f);
        break;
    case IT_TONGUE:
        GfxDraw(MESH_SPHERE, MatLocal(B, V3(0, 0, 0), V3(0, 0.3f*sinf(t*6), 0), V3(0.3f, 0.12f, 0.75f)), (Color){ 255, 105, 150, 255 }, 0, PAT_GOO, 0.9f);
        break;
    }
    if (it->bolted) {
        float glow = 0.6f + 0.4f*sinf(t*5);
        for (int k = 0; k < 3; k++) {
            float a = k*2.1f;
            Vector3 c = Vector3Add(it->pos, V3(cosf(a)*it->radius, -it->radius*0.6f, sinf(a)*it->radius));
            GfxDraw(MESH_CUBE, MatTRS(c, V3(0, a, 0), V3(0.18f, 0.25f, 0.18f)), (Color){ 255, 140, 30, 255 }, glow, PAT_NONE, 0);
        }
    }
}

static void DrawPlayer(void)
{
    Matrix B = BodyMatrix();
    float t = worldTime;
    Color suit = { 232, 232, 226, 255 }, grey = { 110, 112, 120, 255 }, orange = { 240, 120, 30, 255 };
    float o2f = pl.o2/100;
    Color strip = o2f > 0.3f ? (Color){ 60, 220, 255, 255 } : (Color){ 255, 50, 40, 255 };
    // torso, backpack, helmet and visor
    GfxDraw(MESH_SPHERE, MatLocal(B, V3(0, 0.1f, 0), V3(0, 0, 0), V3(0.43f, 0.52f, 0.33f)), suit, 0, PAT_NONE, 0.3f);
    GfxDraw(MESH_CUBE, MatLocal(B, V3(0, 0.22f, -0.36f), V3(0, 0, 0), V3(0.62f, 0.72f, 0.3f)), grey, 0, PAT_PANEL, 0.4f);
    GfxDraw(MESH_CUBE, MatLocal(B, V3(0, 0.22f, -0.52f), V3(0, 0, 0), V3(0.1f, 0.5f, 0.03f)), strip, 1, PAT_NONE, 0);
    GfxDraw(MESH_SPHERE, MatLocal(B, V3(0, 0.78f, 0.02f), V3(0, 0, 0), V3(0.33f, 0.33f, 0.33f)), suit, 0, PAT_NONE, 0.5f);
    GfxDraw(MESH_SPHERE, MatLocal(B, V3(0, 0.78f, 0.17f), V3(0, 0, 0), V3(0.26f, 0.21f, 0.2f)), (Color){ 255, 190, 80, 255 }, 0, PAT_VISOR, 1);
    GfxDraw(MESH_TORUS, MatLocal(B, V3(0, 0.5f, 0.0f), V3(PI/2, 0, 0), V3(0.3f, 0.3f, 0.5f)), grey, 0, PAT_NONE, 0.5f);
    GfxSeg(Vector3Transform(V3(0.16f, 1.0f, -0.1f), B), Vector3Transform(V3(0.22f, 1.35f, -0.2f), B), 0.02f, grey, 0, PAT_NONE, 0.5f);
    GfxDraw(MESH_SPHERE_LO, MatLocal(B, V3(0.22f, 1.36f, -0.2f), V3(0, 0, 0), V3(0.05f, 0.05f, 0.05f)), (Color){ 255, 40, 30, 255 }, fmodf(t, 1.0f) < 0.5f ? 1.0f : 0.2f, PAT_NONE, 0);
    // nozzles
    for (int s = -1; s <= 1; s += 2)
        GfxDraw(MESH_CONE, MatLocal(B, V3(s*0.22f, -0.18f, -0.42f), V3(PI, 0, 0), V3(0.09f, 0.18f, 0.09f)), (Color){ 60, 60, 66, 255 }, pl.thrustVis*0.5f, PAT_NONE, 0.7f);
    // arms: the right one holds the torch forward
    float sw = sinf(t*1.3f)*0.08f;
    for (int s = -1; s <= 1; s += 2) {
        Vector3 sh = V3(s*0.42f, 0.38f, 0);
        Vector3 el = s < 0 ? V3(-0.5f, 0.12f, 0.3f) : V3(0.55f, 0.0f + sw, 0.1f - pl.lean*0.2f);
        Vector3 ha = s < 0 ? V3(-0.42f, 0.0f, 0.55f) : V3(0.5f, -0.3f + sw, 0.25f - pl.lean*0.3f);
        GfxSeg(Vector3Transform(sh, B), Vector3Transform(el, B), 0.11f, suit, 0, PAT_NONE, 0.3f);
        GfxSeg(Vector3Transform(el, B), Vector3Transform(ha, B), 0.1f, suit, 0, PAT_NONE, 0.3f);
        GfxBall(Vector3Transform(ha, B), 0.11f, grey, 0, PAT_NONE, 0.4f);
    }
    GfxSeg(Vector3Transform(V3(-0.42f, 0.0f, 0.45f), B), Vector3Transform(V3(-0.42f, 0.03f, 0.85f), B), 0.06f, orange, pl.cutting ? 0.9f : 0.1f, PAT_NONE, 0.6f);
    // legs dangle in zero-g
    for (int s = -1; s <= 1; s += 2) {
        float k = sinf(t*1.1f + s)*0.12f - pl.lean*0.25f;
        Vector3 hip = V3(s*0.18f, -0.3f, 0), knee = V3(s*0.22f, -0.75f, 0.12f + k), foot = V3(s*0.2f, -1.15f, -0.05f + k*1.5f);
        GfxSeg(Vector3Transform(hip, B), Vector3Transform(knee, B), 0.13f, suit, 0, PAT_NONE, 0.3f);
        GfxSeg(Vector3Transform(knee, B), Vector3Transform(foot, B), 0.12f, suit, 0, PAT_NONE, 0.3f);
        GfxDraw(MESH_CUBE, MatLocal(B, Vector3Add(foot, V3(0, -0.04f, 0.08f)), V3(0, 0, 0), V3(0.18f, 0.12f, 0.3f)), grey, 0, PAT_NONE, 0.4f);
    }
}

static void DrawTether(Vector3 a, Vector3 b, float phase)
{
    Vector3 d = Vector3Subtract(b, a);
    float len = Vector3Length(d);
    if (len < 0.01f) return;
    Vector3 dir = Vector3Scale(d, 1/len);
    Vector3 ref = fabsf(dir.y) < 0.9f ? V3(0, 1, 0) : V3(1, 0, 0);
    Vector3 s1 = Vector3Normalize(Vector3CrossProduct(dir, ref));
    Vector3 s2 = Vector3CrossProduct(dir, s1);
    const int N = 16;
    Vector3 prev = a;
    for (int i = 1; i <= N; i++) {
        float u = (float)i/N;
        float env = sinf(u*PI)*fminf(0.35f, len*0.04f);
        Vector3 p = Vector3Add(a, Vector3Scale(d, u));
        p = Vector3Add(p, Vector3Scale(s1, sinf(u*9 + phase*6)*env));
        p = Vector3Add(p, Vector3Scale(s2, cosf(u*7 + phase*5)*env*0.6f));
        GfxSeg(prev, p, 0.035f, (Color){ 80, 230, 255, 255 }, 1, PAT_NONE, 0);
        prev = p;
    }
}

static void SetupLights(Vector3 view)
{
    GfxClearLights();
    Vector3 lamp = Vector3Add(Vector3Add(pl.pos, Vector3Scale(pl.bodyFwd, 2.2f)), V3(0, 0.8f, 0));
    GfxAddLight(lamp, V3(1.0f, 1.0f, 1.1f));
    if (pl.cutting && pl.cutHit) {
        float fl = Rf(0.7f, 1.3f);
        GfxAddLight(target.point, V3(3.0f*fl, 1.4f*fl, 0.5f*fl));
    }
    float bayPulse = 0.8f + 0.2f*sinf(worldTime*3);
    GfxAddLight(BAY_POS, V3(0.3f*bayPulse, 2.0f*bayPulse, 0.7f*bayPulse));
    float fl = (rand() % 10) < 7 ? 1.0f : 0.2f;
    GfxAddLight(BREACH_LIGHT, V3(2.2f*fl, 0.2f*fl, 0.15f*fl));
    if (fmodf(worldTime, 1.6f) < 0.8f) GfxAddLight(ANTENNA_TIP, V3(1.6f, 0.15f, 0.1f));
    // nearest glowing core and nearest creatures
    float best = 1e9f; int bi = -1;
    for (int i = 0; i < itemCount; i++) {
        if (!items[i].active || items[i].type != IT_CORE) continue;
        float d = Vector3Distance(items[i].pos, view);
        if (d < best) { best = d; bi = i; }
    }
    if (bi >= 0) GfxAddLight(items[bi].pos, V3(0.3f, 1.3f, 1.8f));
    best = 1e9f;
    int gi = -1;
    for (int i = 0; i < gloopCount; i++) {
        if (gloops[i].state == G_DEAD || gloops[i].state == G_LATCHED) continue;
        float d = Vector3Distance(gloops[i].pos, view);
        if (d < best) { best = d; gi = i; }
    }
    if (gi >= 0) {
        Color c = gloops[gi].col;
        GfxAddLight(gloops[gi].pos, V3(c.r/255.0f*0.9f, c.g/255.0f*0.9f, c.b/255.0f*0.9f));
    }
}

void WorldDraw(Camera3D view, bool drawPlayer)
{
    SetupLights(view.position);
    GfxBeginScene(view, worldTime);
    float t = worldTime;
    for (int i = 0; i < partCount; i++) {
        Part *p = &parts[i];
        float em = p->emissive;
        if (p->blink == 1) em = fmodf(t, 1.6f) < 0.8f ? 1.0f : 0.05f;
        else if (p->blink == 2) em = (sinf(t*23 + i) > 0.3f || sinf(t*1.7f + i) > 0) ? 1.0f : 0.1f;
        GfxDraw(p->mesh, p->m, p->c, em, p->pat, p->gloss);
    }
    // bay ring spins and pulses
    float pulse = 0.6f + 0.4f*sinf(t*3);
    GfxDraw(MESH_TORUS, MatTRS(BAY_POS, V3(0, 0, t*0.8f), V3(2.6f, 2.6f, 2.6f)), (Color){ 60, 255, 130, 255 }, pulse, PAT_NONE, 0);
    GfxDraw(MESH_TORUS, MatTRS(BAY_POS, V3(0, 0, -t*0.5f), V3(3.4f, 3.4f, 3.4f)), (Color){ 60, 255, 130, 255 }, pulse*0.6f, PAT_NONE, 0);

    for (int i = 0; i < itemCount; i++) if (items[i].active) DrawItem(&items[i]);
    Vector3 lookAt = drawPlayer ? pl.pos : view.position;
    for (int i = 0; i < gloopCount; i++) {
        if (gloops[i].state == G_DEAD) continue;
        DrawGlooper(&gloops[i], gloops[i].state == G_LATCHED ? pl.pos : lookAt);
    }
    if (drawPlayer) {
        DrawPlayer();
        for (int i = 0; i < MAX_TETHERS; i++) {
            if (!tethers[i].active) continue;
            DrawTether(PackPos(), items[tethers[i].item].pos, t + i);
        }
        if (pl.cutting) {
            Vector3 h = Vector3Transform(V3(-0.42f, 0.03f, 0.85f), BodyMatrix());
            GfxSeg(h, pl.cutEnd, 0.03f + Rf(0, 0.02f), (Color){ 255, 170, 60, 255 }, 1, PAT_NONE, 0);
            GfxSeg(h, pl.cutEnd, 0.012f, (Color){ 255, 255, 220, 255 }, 1, PAT_NONE, 0);
        }
    }
    ParticlesDraw(view);
    GfxEndScene();
}

// ------------------------------------------------------------------ HUD
static void DrawMarker(Camera3D c, Vector3 world, const char *label, Color col, bool pulse)
{
    int sw = GetScreenWidth(), sh = GetScreenHeight();
    Vector3 f = Vector3Normalize(Vector3Subtract(c.target, c.position));
    Vector3 to = Vector3Subtract(world, c.position);
    bool behind = Vector3DotProduct(to, f) < 0;
    Vector2 p = GetWorldToScreen(world, c);
    if (behind) { p.x = sw - p.x; p.y = sh - p.y; }
    float margin = 60;
    bool off = behind || p.x < margin || p.x > sw - margin || p.y < margin || p.y > sh - margin;
    if (off) {
        Vector2 ctr = { sw/2.0f, sh/2.0f };
        Vector2 d = Vector2Subtract(p, ctr);
        if (behind && Vector2Length(d) < 1) d = (Vector2){ 0, 1 };
        float sx = (sw/2.0f - margin)/fmaxf(fabsf(d.x), 1e-3f), sy = (sh/2.0f - margin)/fmaxf(fabsf(d.y), 1e-3f);
        float s = fminf(sx, sy);
        p = Vector2Add(ctr, Vector2Scale(d, s));
    }
    float a = pulse ? 0.6f + 0.4f*sinf(GetTime()*6) : 1.0f;
    Color cc = Fade(col, a);
    float r = 11;
    DrawPoly(p, 4, r, 45, Fade(cc, 0.25f));
    DrawPolyLinesEx(p, 4, r, 45, 2.5f, cc);
    char buf[64];
    snprintf(buf, sizeof buf, "%s %dm", label, (int)Vector3Length(to));
    Vector2 m = UiMeasure(fontMono, buf, 18);
    float tx = Clamp(p.x - m.x/2, 8, sw - m.x - 8);
    UiText(fontMono, buf, tx, p.y + 16, 18, cc);
}

void WorldDrawHUD(Camera3D c)
{
    int sw = GetScreenWidth(), sh = GetScreenHeight();
    float S = sh/900.0f;
    if (S < 0.7f) S = 0.7f;
    float t = (float)GetTime();
    if (dead || won) {
        float a = Clamp((dead ? deadT : winT)*0.6f, 0, 1);
        const char *msg = dead ? "OXYGEN DEPLETED" : "SHIFT COMPLETE";
        UiTextCentered(fontTitle, msg, sw/2.0f, sh*0.4f, 64*S, Fade(dead ? (Color){ 255, 80, 70, 255 } : (Color){ 90, 255, 150, 255 }, a));
    }

    // oxygen
    float o2f = pl.o2/100;
    Color o2c = o2f > 0.35f ? (Color){ 90, 220, 255, 255 } : (o2f > 0.15f ? (Color){ 255, 190, 60, 255 } : (Color){ 255, 70, 60, 255 });
    if (o2f < 0.15f && fmodf(t, 0.5f) < 0.25f) o2c = (Color){ 255, 255, 255, 255 };
    Rectangle o2r = { 28*S, 28*S, 360*S, 74*S };
    UiPanel(o2r, (Color){ 6, 12, 22, 170 }, Fade(o2c, 0.6f));
    UiText(fontMono, "O2", o2r.x + 14*S, o2r.y + 10*S, 22*S, o2c);
    char buf[128];
    snprintf(buf, sizeof buf, "%3d%%", (int)ceilf(pl.o2));
    UiText(fontTitle, buf, o2r.x + o2r.width - 92*S, o2r.y + 4*S, 32*S, o2c);
    UiBar((Rectangle){ o2r.x + 14*S, o2r.y + 42*S, o2r.width - 28*S, 18*S }, o2f, o2c, (Color){ 30, 40, 55, 220 });
    snprintf(buf, sizeof buf, "SHIP RESERVE %d", (int)stats.reserve);
    UiText(fontMono, buf, o2r.x + 50*S, o2r.y + 12*S, 16*S, (Color){ 160, 180, 200, 220 });

    // credits and tethers
    Rectangle cr = { sw - 340*S, 28*S, 312*S, 74*S };
    bool met = stats.credits >= stats.quota;
    Color cc = met ? (Color){ 90, 255, 150, 255 } : (Color){ 255, 210, 90, 255 };
    UiPanel(cr, (Color){ 6, 12, 22, 170 }, Fade(cc, 0.6f));
    snprintf(buf, sizeof buf, "$%d / $%d", stats.credits, stats.quota);
    UiText(fontTitle, buf, cr.x + 14*S, cr.y + 8*S, 30*S, cc);
    UiText(fontMono, "TETHERS", cr.x + 14*S, cr.y + 46*S, 16*S, (Color){ 160, 180, 200, 220 });
    for (int i = 0; i < MAX_TETHERS; i++) {
        Vector2 p = { cr.x + 110*S + i*24*S, cr.y + 54*S };
        if (i < TetherCount()) DrawCircleV(p, 8*S, (Color){ 80, 230, 255, 255 });
        else DrawCircleLinesV(p, 8*S, (Color){ 80, 230, 255, 160 });
    }
    snprintf(buf, sizeof buf, "%.1f m/s", Vector3Length(pl.vel));
    UiText(fontMono, buf, cr.x + cr.width - 100*S, cr.y + 46*S, 16*S, (Color){ 160, 180, 200, 220 });

    // crosshair and target readout
    Vector2 ctr = { sw/2.0f, sh/2.0f };
    Color ch = (Color){ 220, 240, 255, 200 };
    if (target.kind == TGT_ITEM || target.kind == TGT_GLOOP) ch = (Color){ 255, 220, 120, 255 };
    DrawCircleLinesV(ctr, 9*S, ch);
    DrawCircleV(ctr, 2*S, ch);
    float dist = Vector3Distance(target.point, pl.pos);
    if (target.kind == TGT_ITEM && dist < 30) {
        Item *it = &items[target.index];
        const char *name = ItemName(it);
        UiTextCentered(fontUI, name, ctr.x, ctr.y + 24*S, 22*S, (Color){ 255, 235, 180, 255 });
        if (it->type == IT_O2) snprintf(buf, sizeof buf, "+30 O2   fly into it to grab");
        else if (it->type == IT_BOX && it->awake) snprintf(buf, sizeof buf, "HOLD LMB TO TORCH IT");
        else if (it->bolted) snprintf(buf, sizeof buf, "BOLTED   hold LMB to cut  ($%d)", it->value);
        else if (it->tether >= 0) snprintf(buf, sizeof buf, "TETHERED   haul it to the green bay");
        else if (dist > TETHER_RANGE) snprintf(buf, sizeof buf, "too far for a tether (%dm)", (int)dist);
        else if (it->type == IT_BOX) snprintf(buf, sizeof buf, "RMB tether   value: ???");
        else snprintf(buf, sizeof buf, "RMB tether   $%d", it->value);
        UiTextCentered(fontMono, buf, ctr.x, ctr.y + 50*S, 17*S, (Color){ 200, 220, 240, 230 });
        if (it->bolted && it->cut > 0)
            DrawRing(ctr, 14*S, 19*S, -90, -90 + 360*it->cut, 40, (Color){ 255, 160, 50, 255 });
        if (it->type == IT_BOX && it->awake)
            DrawRing(ctr, 14*S, 19*S, -90, -90 + 360*fmaxf(0, it->hp), 40, (Color){ 255, 90, 160, 255 });
    } else if (target.kind == TGT_GLOOP && dist < 30) {
        Glooper *g = &gloops[target.index];
        UiTextCentered(fontUI, "GLOOPER", ctr.x, ctr.y + 24*S, 22*S, g->col);
        UiTextCentered(fontMono, "hold LMB to pop it", ctr.x, ctr.y + 50*S, 17*S, (Color){ 200, 220, 240, 230 });
        DrawRing(ctr, 14*S, 19*S, -90, -90 + 360*fmaxf(0, g->hp), 40, g->col);
    }

    // hugged warning
    int latched = 0;
    for (int i = 0; i < gloopCount; i++) latched += gloops[i].state == G_LATCHED;
    if (latched && !dead) {
        float a = 0.6f + 0.4f*sinf(t*12);
        snprintf(buf, sizeof buf, "GLOOPER HUG!  MASH SHIFT  %d/5", pl.shakeCount);
        UiTextCentered(fontTitle, buf, ctr.x, sh*0.28f, 34*S, Fade((Color){ 255, 120, 200, 255 }, a));
    }

    // ship marker
    DrawMarker(c, BAY_POS, met ? "RETURN TO BAY" : "BAY", (Color){ 80, 255, 140, 255 }, met || o2f < 0.3f);

    // radio
    if (radioT < radioDur) {
        float a = Clamp(fminf(radioT*4, (radioDur - radioT)*2), 0, 1);
        float w = fminf(760*S, sw - 40.0f), h = 92*S;
        Rectangle rr = { (sw - w)/2, sh - h - 26*S, w, h };
        UiPanel(rr, Fade((Color){ 6, 12, 22, 210 }, a), Fade((Color){ 80, 230, 255, 255 }, a*0.7f));
        float talk = radioT < radioDur - 1 ? 1.0f : 0.0f;
        UiDrawBuddy((Vector2){ rr.x + 46*S, rr.y + h/2 }, 30*S*a, talk, t);
        UiText(fontMono, "BUDDY // TUG-7", rr.x + 92*S, rr.y + 10*S, 15*S, Fade((Color){ 80, 230, 255, 255 }, a));
        int shown = (int)(radioT*55);
        char line[200];
        snprintf(line, sizeof line, "%.*s", shown, radioText);
        // simple word wrap into two lines
        float maxW = w - 110*S;
        char l1[200] = { 0 }, l2[200] = { 0 };
        int split = -1, n = (int)strlen(line);
        for (int i = 0; i < n; i++) {
            if (line[i] != ' ') continue;
            char tmp[200];
            snprintf(tmp, sizeof tmp, "%.*s", i, line);
            if (UiMeasure(fontUI, tmp, 20*S).x > maxW) break;
            split = i;
        }
        if (UiMeasure(fontUI, line, 20*S).x <= maxW || split < 0) snprintf(l1, sizeof l1, "%s", line);
        else { snprintf(l1, sizeof l1, "%.*s", split, line); snprintf(l2, sizeof l2, "%s", line + split + 1); }
        UiText(fontUI, l1, rr.x + 92*S, rr.y + 32*S, 20*S, Fade(WHITE, a));
        UiText(fontUI, l2, rr.x + 92*S, rr.y + 58*S, 20*S, Fade(WHITE, a));
    }

    // control hints for the first seconds
    if (stats.time < 14 && !dead) {
        float a = Clamp(14 - stats.time, 0, 1);
        const char *hint = "WASD thrust   SPACE/CTRL up/down   SHIFT boost   X brake   LMB torch   RMB tether   R reel   E release";
        UiTextCentered(fontMono, hint, sw/2.0f, sh - 150*S, 16*S, Fade((Color){ 200, 220, 240, 255 }, a*0.85f));
    }
}

const RunStats *WorldStats(void) { return &stats; }
float WorldDanger(void) { return dead ? 0.6f : Clamp((35 - pl.o2)/35, 0, 1); }
float WorldHurt(void) { return pl.hurt; }
float WorldFade(void) { return dead ? Clamp((deadT - 1.2f)/2, 0, 1) : (won ? Clamp((winT - 0.8f)/1.4f, 0, 1) : 0); }

void WorldDebugStage(int stage)
{
    if (stage == 1) {
        pl.pos = V3(-6, 3, -24);
        pl.vel = V3(0, 0, 0);
        pl.yaw = 0.25f; pl.pitch = 0.05f;
        pl.bodyFwd = ViewFwd();
        Vector3 f = ViewFwd(), r = ViewRight();
        Glooper *g = &gloops[0];
        g->state = G_CHASE;
        g->pos = Vector3Add(Vector3Add(pl.pos, Vector3Scale(f, 8)), Vector3Add(Vector3Scale(r, 2.0f), V3(0, 1.2f, 0)));
        if (gloopCount > 1) {
            gloops[1].state = G_CHASE;
            gloops[1].pos = Vector3Add(Vector3Add(pl.pos, Vector3Scale(f, 15)), Vector3Add(Vector3Scale(r, -5), V3(0, 3, 0)));
        }
        for (int i = 0; i < itemCount; i++) {
            if (items[i].type == IT_BOX && items[i].mimic) {
                items[i].pos = Vector3Add(Vector3Add(pl.pos, Vector3Scale(f, 10)), Vector3Scale(r, 5.5f));
                items[i].awake = true;
                items[i].hopT = 99;
                break;
            }
        }
        for (int i = 0; i < itemCount; i++) {
            if (items[i].type == IT_BATTERY && !items[i].bolted) {
                items[i].pos = Vector3Add(Vector3Add(pl.pos, Vector3Scale(f, 4)), Vector3Add(Vector3Scale(r, 3.5f), V3(0, -1.5f, 0)));
                tethers[0] = (Tether){ true, i, 6 };
                items[i].tether = 0;
                break;
            }
        }
        camPosSmooth = pl.pos;
        Radio("Uh. Something on the scanner is... giggling?");
    } else if (stage == 2) {
        pl.o2 = 0.2f;
        stats.deathCause = "hugged by a Glooper";
    }
}
