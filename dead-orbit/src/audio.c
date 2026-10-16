// All audio is synthesized at startup or in the stream callback; no sound files needed.
#include "audio.h"
#include "raylib.h"
#include <math.h>
#include <stdlib.h>

#define SR 44100
#define TAU 6.28318530718f

static bool ready;
static Sound sfx[SFX_COUNT];
static AudioStream loopStream;
static volatile float wantThrust, wantCut, wantDanger, wantTension, musicVol = 0.7f;

static unsigned int rngState = 0x12345u;
static float Rnd(void)
{
    rngState ^= rngState << 13; rngState ^= rngState >> 17; rngState ^= rngState << 5;
    return (rngState & 0xffffff)/(float)0x800000 - 1.0f;
}

static float Saw(float ph) { return 2.0f*(ph - floorf(ph + 0.5f)); }

// ------------------------------------------------------------------ one-shot synthesis
static float NoteSaw(float t, float f)
{
    float s = 0;
    for (int k = 1; k <= 6; k++) s += sinf(TAU*f*k*t)/k*(k == 1 ? 1.0f : 0.6f);
    return s*0.5f;
}

static float GenSample(SfxId id, float t, float dur)
{
    float u = t/dur;
    switch (id) {
    case SFX_CLICK:
        return sinf(TAU*1400*t)*expf(-t*90)*0.5f;
    case SFX_SELECT: {
        float f = t < 0.07f ? 660 : 990;
        return tanhf(3*sinf(TAU*f*t))*expf(-fmodf(t, 0.07f)*20)*0.3f;
    }
    case SFX_TETHER: {
        float ph = (200*t + 1800/9.0f*(1 - expf(-t*9)));
        return (sinf(TAU*ph)*0.7f + Rnd()*0.4f*expf(-t*25))*expf(-t*6)*0.5f;
    }
    case SFX_RELEASE: {
        float ph = 500*t - 700*t*t;
        return sinf(TAU*ph)*(1 - u)*0.4f;
    }
    case SFX_DEPOSIT: {
        float s = 0;
        float t2 = t - 0.12f;
        s += (sinf(TAU*1318.5f*t) + 0.4f*sinf(TAU*1318.5f*2.76f*t))*expf(-t*6);
        if (t2 > 0) s += (sinf(TAU*1760*t2) + 0.4f*sinf(TAU*1760*2.76f*t2))*expf(-t2*5);
        return s*0.25f;
    }
    case SFX_POP: {
        float ph = 200*t + 3000*t*t;
        float s = sinf(TAU*ph)*expf(-t*14);
        if (t < 0.02f) s += Rnd()*0.6f;
        return s*0.5f;
    }
    case SFX_GIGGLE: {
        static const float pitch[6] = { 700, 840, 660, 920, 760, 1050 };
        float syl = 0.13f;
        int i = (int)(t/syl);
        if (i > 5) return 0;
        float lt = t - i*syl;
        float env = lt < 0.1f ? sinf(3.14159f*lt/0.1f) : 0;
        float f = pitch[i]*(1 + 0.08f*sinf(TAU*32*t)) - 900*lt;
        return (sinf(TAU*f*lt) + 0.3f*sinf(TAU*2*f*lt))*env*0.3f;
    }
    case SFX_BITE: {
        float s = 0;
        for (int k = 0; k < 2; k++) {
            float lt = t - k*0.12f;
            if (lt > 0) s += (Rnd()*0.8f + sinf(TAU*90*lt))*expf(-lt*40);
        }
        return s*0.45f;
    }
    case SFX_ROAR: {
        float f = 95 - 45*u;
        float am = 0.6f + 0.4f*sinf(TAU*23*t) + 0.3f*Rnd();
        float env = fminf(t/0.1f, 1.0f)*(1 - u);
        return tanhf(2.5f*(Saw(f*t) + 0.5f*Saw(f*1.49f*t))*am)*env*0.45f;
    }
    case SFX_UNBOLT: {
        static const float fr[4] = { 320, 487, 731, 1103 };
        float s = 0;
        for (int k = 0; k < 4; k++) s += sinf(TAU*fr[k]*t)*expf(-t*(6 + k*3));
        if (t < 0.03f) s += Rnd();
        return s*0.22f;
    }
    case SFX_BUMP:
        return (sinf(TAU*70*t)*expf(-t*18) + Rnd()*0.3f*expf(-t*40))*0.6f;
    case SFX_ALARM: {
        float lt = fmodf(t, 0.22f);
        return lt < 0.13f ? (sinf(TAU*880*t) > 0 ? 0.22f : -0.22f) : 0;
    }
    case SFX_O2: {
        float s = Rnd()*0.25f*u*(1 - u)*4;
        if (t > 0.3f) s += sinf(TAU*1046*t)*expf(-(t - 0.3f)*10)*0.4f;
        return s;
    }
    case SFX_SLURP: {
        float ph = 180*t + 500*t*t;
        float wah = 0.5f + 0.5f*sinf(TAU*14*t);
        return (sinf(TAU*ph)*wah + Rnd()*0.25f)*sinf(3.14159f*u)*0.45f;
    }
    case SFX_RADIO: {
        float s = Rnd()*0.06f;
        if (t < 0.04f) s += sinf(TAU*1200*t)*0.3f;
        else if (t > 0.07f && t < 0.11f) s += sinf(TAU*1600*t)*0.3f;
        return s*(1 - u);
    }
    case SFX_DEATH: {
        static const float notes[4] = { 233.1f, 220.0f, 207.7f, 196.0f };
        static const float start[4] = { 0.0f, 0.38f, 0.76f, 1.14f };
        static const float len[4] = { 0.34f, 0.34f, 0.34f, 1.2f };
        for (int k = 3; k >= 0; k--) {
            if (t >= start[k]) {
                float lt = t - start[k];
                if (lt > len[k]) return 0;
                float vib = k == 3 ? 1 + 0.03f*sinf(TAU*5.5f*lt)*fminf(lt*2, 1) : 1;
                float env = fminf(lt/0.04f, 1.0f)*(1 - lt/len[k]);
                float wah = 0.55f + 0.45f*sinf(3.14159f*fminf(lt/len[k], 1));
                return NoteSaw(lt, notes[k]*vib)*env*wah*0.4f;
            }
        }
        return 0;
    }
    case SFX_WIN: {
        static const float notes[4] = { 523.3f, 659.3f, 784.0f, 1046.5f };
        int k = (int)(t/0.11f);
        if (k < 4) {
            float lt = t - k*0.11f;
            return NoteSaw(lt, notes[k])*expf(-lt*8)*0.35f;
        }
        float lt = t - 0.44f;
        return (NoteSaw(lt, 523.3f) + NoteSaw(lt, 659.3f) + NoteSaw(lt, 784.0f))*expf(-lt*2.5f)*0.18f;
    }
    default:
        return 0;
    }
}

static const float sfxDur[SFX_COUNT] = {
    0.05f, 0.16f, 0.35f, 0.25f, 0.8f, 0.3f, 0.8f, 0.32f, 1.1f, 0.55f, 0.3f, 0.45f, 0.55f, 0.5f, 0.3f, 2.4f, 1.3f
};

static Sound MakeSound(SfxId id)
{
    float dur = sfxDur[id];
    int n = (int)(dur*SR);
    short *data = malloc(n*sizeof(short));
    for (int i = 0; i < n; i++) {
        float t = (float)i/SR;
        float s = GenSample(id, t, dur);
        float edge = fminf(1.0f, (n - i)/200.0f);
        s = fmaxf(-1.0f, fminf(1.0f, s*edge));
        data[i] = (short)(s*32000);
    }
    Wave w = { (unsigned int)n, SR, 16, 1, data };
    Sound snd = LoadSoundFromWave(w);
    free(data);
    return snd;
}

// ------------------------------------------------------------------ continuous layers
static void LoopCallback(void *buffer, unsigned int frames)
{
    static double t;
    static float thrust, cut, danger, tension, lp, lp2, beatT;
    short *out = buffer;
    const float dt = 1.0f/SR;
    for (unsigned int i = 0; i < frames; i++) {
        thrust += (wantThrust - thrust)*0.0005f;
        cut += (wantCut - cut)*0.002f;
        danger += (wantDanger - danger)*0.0002f;
        tension += (wantTension - tension)*0.0001f;
        float tf = (float)t;
        // eerie drone: detuned low fifths breathing slowly
        float lfo = 0.6f + 0.4f*sinf(TAU*0.07f*tf);
        float drone = (sinf(TAU*55.0f*tf) + sinf(TAU*55.35f*tf) + 0.6f*sinf(TAU*82.4f*tf) + 0.25f*sinf(TAU*110.6f*tf))*0.08f*lfo;
        float pad = (sinf(TAU*329.6f*tf) + sinf(TAU*349.2f*tf)*tension)*0.02f*(0.5f + 0.5f*sinf(TAU*0.13f*tf));
        float music = (drone + pad*(0.4f + tension))*musicVol;
        // thruster hiss
        float n = Rnd();
        lp += (n - lp)*0.12f;
        float th = lp*thrust*0.9f;
        // cutter buzz
        float buzz = (Saw(115.0f*tf) + 0.5f*Saw(231.0f*tf) + 0.3f*n)*(0.7f + 0.3f*sinf(TAU*40.0f*tf));
        lp2 += (buzz - lp2)*0.3f;
        float cu = lp2*cut*0.22f;
        // heartbeat when air runs low
        float period = 1.0f - 0.45f*danger;
        beatT += dt;
        if (beatT > period) beatT -= period;
        float hb = 0;
        if (danger > 0.02f) {
            hb += sinf(TAU*48.0f*beatT)*expf(-beatT*22.0f);
            float b2 = beatT - 0.18f;
            if (b2 > 0) hb += 0.7f*sinf(TAU*44.0f*b2)*expf(-b2*22.0f);
            hb *= danger*0.9f;
        }
        float s = music + th + cu + hb;
        s = tanhf(s);
        out[i] = (short)(s*30000);
        t += dt;
    }
}

void AudioInit(void)
{
    InitAudioDevice();
    ready = IsAudioDeviceReady();
    if (!ready) return;
    for (int i = 0; i < SFX_COUNT; i++) sfx[i] = MakeSound((SfxId)i);
    SetAudioStreamBufferSizeDefault(2048);
    loopStream = LoadAudioStream(SR, 16, 1);
    SetAudioStreamCallback(loopStream, LoopCallback);
    PlayAudioStream(loopStream);
}

void AudioShutdown(void)
{
    if (!ready) return;
    StopAudioStream(loopStream);
    UnloadAudioStream(loopStream);
    for (int i = 0; i < SFX_COUNT; i++) UnloadSound(sfx[i]);
    CloseAudioDevice();
    ready = false;
}

void AudioSetVolumes(float master, float music)
{
    musicVol = music;
    if (ready) SetMasterVolume(master);
}

void AudioSetLoops(float thrust, float cut, float danger, float tension)
{
    wantThrust = thrust; wantCut = cut; wantDanger = danger; wantTension = tension;
}

void SfxPlay(SfxId id, float volume, float pitch)
{
    if (!ready) return;
    SetSoundVolume(sfx[id], volume);
    SetSoundPitch(sfx[id], pitch);
    PlaySound(sfx[id]);
}
