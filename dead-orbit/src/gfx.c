// Rendering: procedural lit materials, sky, planet, particles and post-processing.
#include "gfx.h"
#include "rlgl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GLSL(src) "#version 330\n" src

static const char *NOISE_GLSL =
    "float hash(vec3 p){ p = fract(p*0.3183099 + 0.1); p *= 17.0; return fract(p.x*p.y*p.z*(p.x+p.y+p.z)); }\n"
    "float noise(vec3 x){ vec3 i = floor(x); vec3 f = fract(x); f = f*f*(3.0-2.0*f);\n"
    "  return mix(mix(mix(hash(i+vec3(0,0,0)), hash(i+vec3(1,0,0)), f.x), mix(hash(i+vec3(0,1,0)), hash(i+vec3(1,1,0)), f.x), f.y),\n"
    "             mix(mix(hash(i+vec3(0,0,1)), hash(i+vec3(1,0,1)), f.x), mix(hash(i+vec3(0,1,1)), hash(i+vec3(1,1,1)), f.x), f.y), f.z); }\n"
    "float fbm(vec3 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++){ s += a*noise(p); p = p*2.03 + vec3(1.7, 9.2, 3.1); a *= 0.5; } return s; }\n";

static const char *MESH_VS = GLSL(
    "in vec3 vertexPosition; in vec2 vertexTexCoord; in vec3 vertexNormal;\n"
    "uniform mat4 mvp; uniform mat4 matModel; uniform mat4 matNormal;\n"
    "out vec3 fragPos; out vec3 fragNormal; out vec2 fragTexCoord; out vec3 localPos;\n"
    "void main(){\n"
    "  fragPos = vec3(matModel*vec4(vertexPosition, 1.0));\n"
    "  fragNormal = normalize(vec3(matNormal*vec4(vertexNormal, 0.0)));\n"
    "  fragTexCoord = vertexTexCoord; localPos = vertexPosition;\n"
    "  gl_Position = mvp*vec4(vertexPosition, 1.0);\n"
    "}\n");

static const char *LIT_FS_HEAD =
    "#version 330\n"
    "in vec3 fragPos; in vec3 fragNormal; in vec2 fragTexCoord; in vec3 localPos;\n"
    "uniform vec4 colDiffuse; uniform vec3 viewPos; uniform vec3 sunDir; uniform float time;\n"
    "uniform float emissive; uniform int pattern; uniform float gloss;\n"
    "uniform int lightCount; uniform vec3 lightPos[8]; uniform vec3 lightColor[8];\n"
    "out vec4 finalColor;\n";

static const char *LIT_FS_BODY =
    "vec2 triUV(vec3 p, vec3 n){ vec3 a = abs(n); return a.x > a.y && a.x > a.z ? p.yz : (a.y > a.z ? p.xz : p.xy); }\n"
    "void main(){\n"
    "  vec3 n = normalize(fragNormal); if (!gl_FrontFacing) n = -n;\n"
    "  vec3 v = normalize(viewPos - fragPos);\n"
    "  vec3 albedo = colDiffuse.rgb; float g = gloss; vec3 extra = vec3(0.0);\n"
    "  if (pattern == 1) {\n"
    "    vec2 uv = triUV(fragPos*0.7, n); vec2 q = abs(fract(uv) - 0.5);\n"
    "    float seam = smoothstep(0.46, 0.5, max(q.x, q.y));\n"
    "    float rivet = step(0.43, q.x)*step(0.43, q.y);\n"
    "    float grime = 0.62 + 0.5*fbm(fragPos*0.9);\n"
    "    float scorch = smoothstep(0.62, 0.8, fbm(fragPos*0.35 + 4.0));\n"
    "    albedo *= grime*(1.0 - 0.5*seam)*(1.0 - 0.6*scorch); albedo += rivet*0.05;\n"
    "    g *= (1.0 - scorch);\n"
    "  } else if (pattern == 2) {\n"
    "    vec2 uv = triUV(fragPos*2.0, n); vec2 q = abs(fract(uv) - 0.5);\n"
    "    float line = smoothstep(0.44, 0.5, max(q.x, q.y));\n"
    "    albedo = mix(vec3(0.03, 0.06, 0.18)*(0.8 + 0.4*noise(fragPos*3.0)), vec3(0.55), line); g = 0.9;\n"
    "  } else if (pattern == 3) {\n"
    "    float c = noise(fragPos*5.0)*0.6 + noise(fragPos*13.0)*0.4;\n"
    "    albedo *= 0.55 + 0.9*c;\n"
    "    n = normalize(n + (vec3(noise(fragPos*9.0), noise(fragPos*9.0 + 3.1), noise(fragPos*9.0 + 7.7)) - 0.5)*0.9);\n"
    "  } else if (pattern == 4) {\n"
    "    float veins = smoothstep(0.55, 0.62, noise(fragPos*2.5 + vec3(0.0, time*0.8, 0.0)));\n"
    "    albedo *= 1.0 - 0.45*veins;\n"
    "    float ndv = max(dot(n, v), 0.0);\n"
    "    extra = albedo*0.55*pow(1.0 - ndv, 2.0) + albedo*0.12;\n"
    "    g = 0.85;\n"
    "  } else if (pattern == 5) {\n"
    "    vec3 r = reflect(-v, n);\n"
    "    vec3 env = mix(vec3(0.05, 0.12, 0.3), vec3(0.01, 0.01, 0.02), smoothstep(-0.3, 0.2, r.y));\n"
    "    env += vec3(1.0, 0.9, 0.7)*pow(max(dot(r, sunDir), 0.0), 60.0)*3.0;\n"
    "    extra = env*albedo*1.6; albedo *= 0.25; g = 1.0;\n"
    "  } else if (pattern == 6) {\n"
    "    float ndv = max(dot(n, v), 0.0);\n"
    "    extra = albedo*(0.35 + 0.6*pow(1.0 - ndv, 2.0)); g = 1.0;\n"
    "  }\n"
    "  float ndl = max(dot(n, sunDir), 0.0);\n"
    "  vec3 sunC = vec3(1.0, 0.95, 0.88)*1.25;\n"
    "  vec3 h = normalize(sunDir + v);\n"
    "  float sp = pow(max(dot(n, h), 0.0), mix(6.0, 120.0, g))*g;\n"
    "  vec3 amb = mix(vec3(0.05, 0.08, 0.14), vec3(0.025, 0.025, 0.035), n.y*0.5 + 0.5);\n"
    "  vec3 col = albedo*(sunC*ndl + amb) + sunC*sp*ndl*0.8;\n"
    "  for (int i = 0; i < lightCount; i++) {\n"
    "    vec3 L = lightPos[i] - fragPos; float d = length(L); L /= d;\n"
    "    float att = 1.0/(1.0 + 0.12*d + 0.06*d*d);\n"
    "    float dl = max(dot(n, L), 0.0);\n"
    "    float sl = pow(max(dot(n, normalize(L + v)), 0.0), mix(6.0, 120.0, g))*g;\n"
    "    col += (albedo*dl + sl*0.6)*lightColor[i]*att;\n"
    "  }\n"
    "  float rim = pow(1.0 - max(dot(n, v), 0.0), 4.0);\n"
    "  col += rim*vec3(0.12, 0.22, 0.45)*(0.4 + 0.6*clamp(-n.y + 0.3, 0.0, 1.0));\n"
    "  col += extra;\n"
    "  col = mix(col, albedo*1.8, emissive);\n"
    "  finalColor = vec4(col, colDiffuse.a);\n"
    "}\n";

static const char *SKY_FS_BODY =
    "void main(){\n"
    "  vec3 d = normalize(fragPos - viewPos);\n"
    "  vec3 col = vec3(0.0);\n"
    "  float n1 = fbm(d*2.2 + 11.0), n2 = fbm(d*4.5 + 3.0);\n"
    "  col += vec3(0.22, 0.06, 0.3)*pow(n1, 3.0)*1.1 + vec3(0.03, 0.12, 0.26)*pow(n2, 4.0)*1.6;\n"
    "  for (int k = 0; k < 2; k++) {\n"
    "    float sc = k == 0 ? 220.0 : 480.0;\n"
    "    vec3 q = d*sc; vec3 cell = floor(q);\n"
    "    float h = hash(cell + float(k)*13.0);\n"
    "    vec3 c = cell + 0.5 + (vec3(hash(cell + 1.3), hash(cell + 5.1), hash(cell + 9.7)) - 0.5)*0.6;\n"
    "    float dist = length(q - c);\n"
    "    float tw = 0.75 + 0.25*sin(time*(2.0 + h*5.0) + h*50.0);\n"
    "    float th = k == 0 ? 0.992 : 0.985;\n"
    "    col += step(th, h)*smoothstep(0.45, 0.0, dist)*tw*mix(vec3(0.7, 0.8, 1.0), vec3(1.0, 0.85, 0.7), hash(cell + 2.2))*(k == 0 ? 1.6 : 0.8);\n"
    "  }\n"
    "  float s = max(dot(d, sunDir), 0.0);\n"
    "  col += vec3(1.0, 0.9, 0.75)*(pow(s, 900.0)*12.0 + pow(s, 60.0)*0.35 + pow(s, 8.0)*0.05);\n"
    "  finalColor = vec4(col, 1.0);\n"
    "}\n";

static const char *PLANET_FS_BODY =
    "void main(){\n"
    "  vec3 n = normalize(fragNormal); vec3 p = normalize(localPos);\n"
    "  vec3 v = normalize(viewPos - fragPos);\n"
    "  float w = fbm(p*3.0);\n"
    "  float land = fbm(p*5.0 + w*1.6);\n"
    "  float lm = smoothstep(0.5, 0.53, land);\n"
    "  vec3 landc = mix(vec3(0.09, 0.16, 0.05), vec3(0.42, 0.33, 0.2), smoothstep(0.55, 0.72, land));\n"
    "  vec3 alb = mix(vec3(0.01, 0.045, 0.13), landc, lm);\n"
    "  alb = mix(alb, vec3(0.9), smoothstep(0.82, 0.9, abs(p.y) + w*0.1));\n"
    "  float cw = fbm(p*4.0 + 5.0);\n"
    "  float cl = smoothstep(0.5, 0.78, fbm(p*9.0 + vec3(time*0.004, 0.0, 0.0) + cw*2.0));\n"
    "  alb = mix(alb, vec3(0.95), cl);\n"
    "  float dif = max(dot(n, sunDir), 0.0);\n"
    "  vec3 col = alb*dif*vec3(1.0, 0.97, 0.92)*1.15;\n"
    "  col += vec3(1.0, 0.9, 0.7)*pow(max(dot(reflect(-sunDir, n), v), 0.0), 50.0)*(1.0 - lm)*(1.0 - cl)*dif*0.7;\n"
    "  float city = step(0.93, hash(floor(p*220.0)))*lm*(1.0 - cl);\n"
    "  col += vec3(1.0, 0.55, 0.2)*city*smoothstep(0.08, -0.2, dot(n, sunDir))*0.9;\n"
    "  float ndv = max(dot(n, v), 0.0);\n"
    "  col = mix(col, vec3(0.3, 0.55, 1.0)*(dif*1.3 + 0.02), pow(1.0 - ndv, 3.0)*0.6);\n"
    "  finalColor = vec4(col, 1.0);\n"
    "}\n";

static const char *HALO_FS_BODY =
    "void main(){\n"
    "  vec3 n = normalize(fragNormal); vec3 v = normalize(viewPos - fragPos);\n"
    "  float ndv = max(dot(n, v), 0.0);\n"
    "  float a = smoothstep(0.0, 0.28, ndv)*(1.0 - smoothstep(0.28, 0.55, ndv));\n"
    "  float lit = clamp(dot(n, sunDir) + 0.35, 0.0, 1.0);\n"
    "  finalColor = vec4(vec3(0.3, 0.55, 1.0)*a*lit*0.7, 1.0);\n"
    "}\n";

static const char *BRIGHT_FS = GLSL(
    "in vec2 fragTexCoord; uniform sampler2D texture0; uniform vec2 texel; out vec4 finalColor;\n"
    "void main(){\n"
    "  vec3 c = vec3(0.0);\n"
    "  for (int x = -1; x <= 1; x += 2) for (int y = -1; y <= 1; y += 2) c += texture(texture0, fragTexCoord + vec2(x, y)*texel*1.5).rgb;\n"
    "  c *= 0.25;\n"
    "  float l = max(c.r, max(c.g, c.b));\n"
    "  finalColor = vec4(c*smoothstep(0.55, 1.0, l)*1.4, 1.0);\n"
    "}\n");

static const char *BLUR_FS = GLSL(
    "in vec2 fragTexCoord; uniform sampler2D texture0; uniform vec2 dir; out vec4 finalColor;\n"
    "void main(){\n"
    "  float w[5] = float[](0.227, 0.194, 0.121, 0.054, 0.016);\n"
    "  vec3 c = texture(texture0, fragTexCoord).rgb*w[0];\n"
    "  for (int i = 1; i < 5; i++) {\n"
    "    c += texture(texture0, fragTexCoord + dir*float(i)*1.6).rgb*w[i];\n"
    "    c += texture(texture0, fragTexCoord - dir*float(i)*1.6).rgb*w[i];\n"
    "  }\n"
    "  finalColor = vec4(c, 1.0);\n"
    "}\n");

static const char *COMPOSITE_FS = GLSL(
    "in vec2 fragTexCoord; uniform sampler2D texture0; uniform sampler2D bloomTex;\n"
    "uniform vec2 resolution; uniform float time; uniform float danger; uniform float hurt; uniform float fade; uniform float bloomOn;\n"
    "out vec4 finalColor;\n"
    "float h21(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233)))*43758.5453); }\n"
    "void main(){\n"
    "  vec2 uv = fragTexCoord; vec2 cc = uv - 0.5;\n"
    "  float wob = danger*0.004*sin(time*3.0 + uv.y*12.0);\n"
    "  uv.x += wob;\n"
    "  float ca = (0.0012 + danger*0.007 + hurt*0.01)*length(cc);\n"
    "  vec2 dir = normalize(cc + 1e-5);\n"
    "  vec3 col = vec3(texture(texture0, uv + dir*ca).r, texture(texture0, uv).g, texture(texture0, uv - dir*ca).b);\n"
    "  vec3 bl = texture(bloomTex, uv).rgb;\n"
    "  col += bl*1.1*bloomOn;\n"
    "  col = col/(1.0 + col*0.15)*1.12;\n"
    "  float vig = 1.0 - dot(cc, cc)*(0.9 + danger*1.6);\n"
    "  col *= clamp(vig, 0.0, 1.0);\n"
    "  float pulse = danger*(0.5 + 0.5*sin(time*6.0));\n"
    "  col = mix(col, col*vec3(1.4, 0.35, 0.3) + vec3(0.08, 0.0, 0.0), pulse*0.55*smoothstep(0.15, 0.7, length(cc)));\n"
    "  col += vec3(0.6, 0.05, 0.05)*hurt*0.5;\n"
    "  col += (h21(uv*resolution + fract(time)*100.0) - 0.5)*0.035;\n"
    "  col *= 1.0 - fade;\n"
    "  finalColor = vec4(col, 1.0);\n"
    "}\n");

// ------------------------------------------------------------------ state
static Mesh meshes[MESH_COUNT];
static Material litMat, skyMat, planetMat, haloMat;
static Shader litShader, skyShader, planetShader, haloShader, brightShader, blurShader, compShader;
static int locLit[8], locSky[3], locPlanet[3], locHalo[2];
static int locEmissive, locPattern, locGloss, locLightCount, locLightPos, locLightColor;
static int locBrightTexel, locBlurDir, locCompBloom, locCompRes, locCompTime, locCompDanger, locCompHurt, locCompFade, locCompBloomOn;
static RenderTexture2D rtScene, rtBloomA, rtBloomB;
static int rtW, rtH;
static Texture2D dotTex;
static Vector3 lightPos[GFX_MAX_LIGHTS], lightCol[GFX_MAX_LIGHTS];
static int lightCount;
static float gTime;
static Vector3 sunDir;

typedef struct { Vector3 pos, vel; float life, maxLife, size; Color c; } Particle;
static Particle parts[GFX_MAX_PARTICLES];
static int partCount;

static char *Concat3(const char *a, const char *b, const char *c)
{
    size_t la = strlen(a), lb = strlen(b), lc = strlen(c);
    char *s = malloc(la + lb + lc + 1);
    memcpy(s, a, la); memcpy(s + la, b, lb); memcpy(s + la + lb, c, lc);
    s[la + lb + lc] = 0;
    return s;
}

static Shader LoadMeshShader(const char *body)
{
    char *fs = Concat3(LIT_FS_HEAD, NOISE_GLSL, body);
    Shader s = LoadShaderFromMemory(MESH_VS, fs);
    free(fs);
    s.locs[SHADER_LOC_MATRIX_MODEL] = GetShaderLocation(s, "matModel");
    s.locs[SHADER_LOC_MATRIX_NORMAL] = GetShaderLocation(s, "matNormal");
    return s;
}

static void LoadTargets(void)
{
    rtW = GetScreenWidth(); rtH = GetScreenHeight();
    if (rtW < 64) rtW = 64;
    if (rtH < 64) rtH = 64;
    rtScene = LoadRenderTexture(rtW, rtH);
    rtBloomA = LoadRenderTexture(rtW/4, rtH/4);
    rtBloomB = LoadRenderTexture(rtW/4, rtH/4);
    SetTextureFilter(rtScene.texture, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(rtBloomA.texture, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(rtBloomB.texture, TEXTURE_FILTER_BILINEAR);
}

static void UnloadTargets(void)
{
    UnloadRenderTexture(rtScene);
    UnloadRenderTexture(rtBloomA);
    UnloadRenderTexture(rtBloomB);
}

void GfxInit(void)
{
    meshes[MESH_CUBE] = GenMeshCube(1, 1, 1);
    meshes[MESH_SPHERE] = GenMeshSphere(1, 32, 32);
    meshes[MESH_SPHERE_LO] = GenMeshSphere(1, 12, 14);
    meshes[MESH_CYL] = GenMeshCylinder(1, 1, 24);
    meshes[MESH_CONE] = GenMeshCone(1, 1, 16);
    meshes[MESH_TORUS] = GenMeshTorus(0.12f, 2.0f, 48, 12);

    litShader = LoadMeshShader(LIT_FS_BODY);
    skyShader = LoadMeshShader(SKY_FS_BODY);
    planetShader = LoadMeshShader(PLANET_FS_BODY);
    haloShader = LoadMeshShader(HALO_FS_BODY);

    const char *common[3] = { "viewPos", "sunDir", "time" };
    for (int i = 0; i < 3; i++) {
        locLit[i] = GetShaderLocation(litShader, common[i]);
        locSky[i] = GetShaderLocation(skyShader, common[i]);
        locPlanet[i] = GetShaderLocation(planetShader, common[i]);
    }
    locHalo[0] = GetShaderLocation(haloShader, "viewPos");
    locHalo[1] = GetShaderLocation(haloShader, "sunDir");
    locEmissive = GetShaderLocation(litShader, "emissive");
    locPattern = GetShaderLocation(litShader, "pattern");
    locGloss = GetShaderLocation(litShader, "gloss");
    locLightCount = GetShaderLocation(litShader, "lightCount");
    locLightPos = GetShaderLocation(litShader, "lightPos");
    locLightColor = GetShaderLocation(litShader, "lightColor");

    litMat = LoadMaterialDefault(); litMat.shader = litShader;
    skyMat = LoadMaterialDefault(); skyMat.shader = skyShader;
    planetMat = LoadMaterialDefault(); planetMat.shader = planetShader;
    haloMat = LoadMaterialDefault(); haloMat.shader = haloShader;

    brightShader = LoadShaderFromMemory(NULL, BRIGHT_FS);
    blurShader = LoadShaderFromMemory(NULL, BLUR_FS);
    compShader = LoadShaderFromMemory(NULL, COMPOSITE_FS);
    locBrightTexel = GetShaderLocation(brightShader, "texel");
    locBlurDir = GetShaderLocation(blurShader, "dir");
    locCompBloom = GetShaderLocation(compShader, "bloomTex");
    locCompRes = GetShaderLocation(compShader, "resolution");
    locCompTime = GetShaderLocation(compShader, "time");
    locCompDanger = GetShaderLocation(compShader, "danger");
    locCompHurt = GetShaderLocation(compShader, "hurt");
    locCompFade = GetShaderLocation(compShader, "fade");
    locCompBloomOn = GetShaderLocation(compShader, "bloomOn");

    Image img = GenImageGradientRadial(64, 64, 0.0f, WHITE, BLANK);
    dotTex = LoadTextureFromImage(img);
    UnloadImage(img);
    SetTextureFilter(dotTex, TEXTURE_FILTER_BILINEAR);

    sunDir = Vector3Normalize((Vector3){ -0.75f, 0.45f, 0.35f });
    LoadTargets();
}

void GfxUnload(void)
{
    UnloadTargets();
    for (int i = 0; i < MESH_COUNT; i++) UnloadMesh(meshes[i]);
    UnloadShader(litShader); UnloadShader(skyShader); UnloadShader(planetShader); UnloadShader(haloShader);
    UnloadShader(brightShader); UnloadShader(blurShader); UnloadShader(compShader);
    UnloadTexture(dotTex);
}

void GfxClearLights(void) { lightCount = 0; }

void GfxAddLight(Vector3 pos, Vector3 color)
{
    if (lightCount >= GFX_MAX_LIGHTS) return;
    lightPos[lightCount] = pos;
    lightCol[lightCount] = color;
    lightCount++;
}

Matrix MatTRS(Vector3 t, Vector3 e, Vector3 s)
{
    return MatrixMultiply(MatrixMultiply(MatrixScale(s.x, s.y, s.z), MatrixRotateXYZ(e)), MatrixTranslate(t.x, t.y, t.z));
}

Matrix MatBasis(Vector3 r, Vector3 u, Vector3 f, Vector3 p)
{
    Matrix m = { 0 };
    m.m0 = r.x; m.m1 = r.y; m.m2 = r.z;
    m.m4 = u.x; m.m5 = u.y; m.m6 = u.z;
    m.m8 = f.x; m.m9 = f.y; m.m10 = f.z;
    m.m12 = p.x; m.m13 = p.y; m.m14 = p.z; m.m15 = 1;
    return m;
}

Matrix MatLocal(Matrix parent, Vector3 t, Vector3 e, Vector3 s)
{
    return MatrixMultiply(MatTRS(t, e, s), parent);
}

void GfxDraw(MeshKind kind, Matrix m, Color c, float emissive, Pattern pat, float gloss)
{
    int p = (int)pat;
    SetShaderValue(litShader, locEmissive, &emissive, SHADER_UNIFORM_FLOAT);
    SetShaderValue(litShader, locPattern, &p, SHADER_UNIFORM_INT);
    SetShaderValue(litShader, locGloss, &gloss, SHADER_UNIFORM_FLOAT);
    litMat.maps[MATERIAL_MAP_DIFFUSE].color = c;
    DrawMesh(meshes[kind], litMat, m);
}

void GfxSeg(Vector3 a, Vector3 b, float r, Color c, float emissive, Pattern pat, float gloss)
{
    Vector3 d = Vector3Subtract(b, a);
    float len = Vector3Length(d);
    if (len < 1e-4f) return;
    Vector3 up = Vector3Scale(d, 1.0f/len);
    Vector3 ref = fabsf(up.y) < 0.95f ? (Vector3){ 0, 1, 0 } : (Vector3){ 1, 0, 0 };
    Vector3 right = Vector3Normalize(Vector3CrossProduct(ref, up));
    Vector3 fwd = Vector3CrossProduct(right, up);
    Matrix m = MatBasis(Vector3Scale(right, r), Vector3Scale(up, len), Vector3Scale(fwd, r), a);
    GfxDraw(MESH_CYL, m, c, emissive, pat, gloss);
}

void GfxBall(Vector3 p, float r, Color c, float emissive, Pattern pat, float gloss)
{
    GfxDraw(r > 0.6f ? MESH_SPHERE : MESH_SPHERE_LO, MatTRS(p, (Vector3){ 0 }, (Vector3){ r, r, r }), c, emissive, pat, gloss);
}

static void SetCommon(Shader s, const int *locs, Vector3 view)
{
    SetShaderValue(s, locs[0], &view, SHADER_UNIFORM_VEC3);
    SetShaderValue(s, locs[1], &sunDir, SHADER_UNIFORM_VEC3);
    SetShaderValue(s, locs[2], &gTime, SHADER_UNIFORM_FLOAT);
}

void GfxBeginScene(Camera3D cam, float time)
{
    if (GetScreenWidth() != rtW || GetScreenHeight() != rtH) {
        UnloadTargets();
        LoadTargets();
    }
    gTime = time;
    SetCommon(litShader, locLit, cam.position);
    SetCommon(skyShader, locSky, cam.position);
    SetCommon(planetShader, locPlanet, cam.position);
    SetShaderValue(haloShader, locHalo[0], &cam.position, SHADER_UNIFORM_VEC3);
    SetShaderValue(haloShader, locHalo[1], &sunDir, SHADER_UNIFORM_VEC3);
    SetShaderValue(litShader, locLightCount, &lightCount, SHADER_UNIFORM_INT);
    if (lightCount > 0) {
        SetShaderValueV(litShader, locLightPos, lightPos, SHADER_UNIFORM_VEC3, lightCount);
        SetShaderValueV(litShader, locLightColor, lightCol, SHADER_UNIFORM_VEC3, lightCount);
    }

    BeginTextureMode(rtScene);
    ClearBackground(BLACK);
    BeginMode3D(cam);

    // Background: sky dome and planet follow the camera so they never get closer.
    rlDisableDepthMask();
    rlDisableDepthTest();
    rlDisableBackfaceCulling();
    DrawMesh(meshes[MESH_SPHERE], skyMat, MatTRS(cam.position, (Vector3){ 0 }, (Vector3){ 900, 900, 900 }));
    rlEnableBackfaceCulling();
    Vector3 pc = Vector3Add(cam.position, (Vector3){ 60.0f, -560.0f, 300.0f });
    DrawMesh(meshes[MESH_SPHERE], planetMat, MatTRS(pc, (Vector3){ 0.3f, time*0.004f, 0.1f }, (Vector3){ 470, 470, 470 }));
    BeginBlendMode(BLEND_ADDITIVE);
    DrawMesh(meshes[MESH_SPHERE], haloMat, MatTRS(pc, (Vector3){ 0 }, (Vector3){ 500, 500, 500 }));
    EndBlendMode();
    rlEnableDepthTest();
    rlEnableDepthMask();
}

void GfxEndScene(void)
{
    EndMode3D();
    EndTextureMode();
}

static void Blit(RenderTexture2D src, int sw, int sh, int dw, int dh)
{
    DrawTexturePro(src.texture, (Rectangle){ 0, 0, (float)sw, (float)-sh }, (Rectangle){ 0, 0, (float)dw, (float)dh }, (Vector2){ 0 }, 0, WHITE);
}

void GfxPresent(float danger, float hurt, float fade, bool bloom)
{
    int bw = rtW/4, bh = rtH/4;
    if (bloom) {
        Vector2 texel = { 1.0f/rtW, 1.0f/rtH };
        BeginTextureMode(rtBloomA);
        BeginShaderMode(brightShader);
        SetShaderValue(brightShader, locBrightTexel, &texel, SHADER_UNIFORM_VEC2);
        Blit(rtScene, rtW, rtH, bw, bh);
        EndShaderMode();
        EndTextureMode();
        for (int pass = 0; pass < 2; pass++) {
            Vector2 dx = { 1.0f/bw, 0 }, dy = { 0, 1.0f/bh };
            BeginTextureMode(rtBloomB);
            BeginShaderMode(blurShader);
            SetShaderValue(blurShader, locBlurDir, &dx, SHADER_UNIFORM_VEC2);
            Blit(rtBloomA, bw, bh, bw, bh);
            EndShaderMode();
            EndTextureMode();
            BeginTextureMode(rtBloomA);
            BeginShaderMode(blurShader);
            SetShaderValue(blurShader, locBlurDir, &dy, SHADER_UNIFORM_VEC2);
            Blit(rtBloomB, bw, bh, bw, bh);
            EndShaderMode();
            EndTextureMode();
        }
    }
    float bloomOn = bloom ? 1.0f : 0.0f;
    Vector2 res = { (float)rtW, (float)rtH };
    BeginShaderMode(compShader);
    SetShaderValueTexture(compShader, locCompBloom, rtBloomA.texture);
    SetShaderValue(compShader, locCompRes, &res, SHADER_UNIFORM_VEC2);
    SetShaderValue(compShader, locCompTime, &gTime, SHADER_UNIFORM_FLOAT);
    SetShaderValue(compShader, locCompDanger, &danger, SHADER_UNIFORM_FLOAT);
    SetShaderValue(compShader, locCompHurt, &hurt, SHADER_UNIFORM_FLOAT);
    SetShaderValue(compShader, locCompFade, &fade, SHADER_UNIFORM_FLOAT);
    SetShaderValue(compShader, locCompBloomOn, &bloomOn, SHADER_UNIFORM_FLOAT);
    Blit(rtScene, rtW, rtH, GetScreenWidth(), GetScreenHeight());
    EndShaderMode();
}

// ------------------------------------------------------------------ particles
void ParticleSpawn(Vector3 pos, Vector3 vel, float life, float size, Color c)
{
    if (partCount >= GFX_MAX_PARTICLES) return;
    parts[partCount++] = (Particle){ pos, vel, life, life, size, c };
}

void ParticlesUpdate(float dt)
{
    for (int i = 0; i < partCount; i++) {
        Particle *p = &parts[i];
        p->life -= dt;
        if (p->life <= 0) { parts[i--] = parts[--partCount]; continue; }
        p->pos = Vector3Add(p->pos, Vector3Scale(p->vel, dt));
        p->vel = Vector3Scale(p->vel, 1.0f - 0.6f*dt);
    }
}

void ParticlesDraw(Camera3D cam)
{
    rlDisableDepthMask();
    BeginBlendMode(BLEND_ADDITIVE);
    for (int i = 0; i < partCount; i++) {
        Particle *p = &parts[i];
        float t = p->life/p->maxLife;
        Color c = p->c;
        c.a = (unsigned char)(c.a*t);
        DrawBillboard(cam, dotTex, p->pos, p->size*(0.6f + 0.6f*(1 - t)), c);
    }
    EndBlendMode();
    rlEnableDepthMask();
}

void ParticlesClear(void) { partCount = 0; }
