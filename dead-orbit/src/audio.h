#ifndef AUDIO_H
#define AUDIO_H

typedef enum {
    SFX_CLICK, SFX_SELECT, SFX_TETHER, SFX_RELEASE, SFX_DEPOSIT, SFX_POP, SFX_GIGGLE,
    SFX_BITE, SFX_ROAR, SFX_UNBOLT, SFX_BUMP, SFX_ALARM, SFX_O2, SFX_SLURP, SFX_RADIO,
    SFX_DEATH, SFX_WIN, SFX_COUNT
} SfxId;

void AudioInit(void);
void AudioShutdown(void);
void AudioSetVolumes(float master, float music);
// Continuous layers, each 0..1: thruster hiss, cutter buzz, low-oxygen heartbeat, creature tension.
void AudioSetLoops(float thrust, float cut, float danger, float tension);
void SfxPlay(SfxId id, float volume, float pitch);

#endif
