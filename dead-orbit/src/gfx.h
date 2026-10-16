#ifndef GFX_H
#define GFX_H

#include "raylib.h"
#include "raymath.h"
#include <stdbool.h>

typedef enum {
    MESH_CUBE, MESH_SPHERE, MESH_SPHERE_LO, MESH_CYL, MESH_CONE, MESH_TORUS, MESH_COUNT
} MeshKind;

typedef enum {
    PAT_NONE,   // plain painted surface
    PAT_PANEL,  // hull plating with seams and grime
    PAT_SOLAR,  // solar cell grid
    PAT_FOIL,   // crinkled gold insulation foil
    PAT_GOO,    // translucent wobbly creature flesh
    PAT_VISOR,  // gold helmet visor reflecting the planet
    PAT_GLASS,  // glowing cockpit glass
} Pattern;

#define GFX_MAX_LIGHTS 8
#define GFX_MAX_PARTICLES 3000

void GfxInit(void);
void GfxUnload(void);

void GfxClearLights(void);
void GfxAddLight(Vector3 pos, Vector3 color);

// Frame: BeginScene binds the offscreen target and draws sky + planet.
void GfxBeginScene(Camera3D cam, float time);
void GfxEndScene(void);
// Composites the scene with bloom and screen effects onto the backbuffer.
void GfxPresent(float danger, float hurt, float fade, bool bloom);

void GfxDraw(MeshKind kind, Matrix m, Color c, float emissive, Pattern pat, float gloss);
void GfxSeg(Vector3 a, Vector3 b, float r, Color c, float emissive, Pattern pat, float gloss);
void GfxBall(Vector3 p, float r, Color c, float emissive, Pattern pat, float gloss);

Matrix MatTRS(Vector3 t, Vector3 eulerRad, Vector3 s);
Matrix MatBasis(Vector3 right, Vector3 up, Vector3 fwd, Vector3 pos);
Matrix MatLocal(Matrix parent, Vector3 t, Vector3 eulerRad, Vector3 s);

// particles
void ParticleSpawn(Vector3 pos, Vector3 vel, float life, float size, Color c);
void ParticlesUpdate(float dt);
void ParticlesDraw(Camera3D cam);
void ParticlesClear(void);

#endif
