# Indie game concepts (C on Linux)

Mockups are 320x180 pixel art scaled 4x. To re-render them, run `python3 make_concepts.py`.

| # | Concept | Core hook | Scope |
|---|---------|-----------|-------|
| 1 | **Lantern Below** | Roguelite mining dive. Your lantern oil is both your light and your health, so mining ore to buy oil is the core loop. | Medium |
| 2 | **Tiny Orbit** | One-shot gravity slingshot puzzles. You get one burn, and the planets do the rest. | Small (best first project) |
| 3 | **Daemon Keeper** | Tower defense inside a Linux box: `cron`, `kill`, `grep`, `oom` towers vs. rogue processes and fork bombs. | Small-medium |
| 4 | **Rootbound** | Farm by day; at night, cut back corrupted vines creeping toward your crops in a small lantern radius. | Medium |

## Suggested stack
- **raylib** (simplest) or **SDL2/SDL3** for window, input, audio
- C99/C11, `make` or CMake, render to a 320x180 texture and scale up
- Ship with AppImage or Flatpak, plus Steam (Proton is unnecessary, as it's native)

---

# 3D concepts

Rendered by `render_3d_concepts.py`, a NumPy raymarcher with soft shadows, ambient occlusion, fog, water reflections and volumetric light. Run `python3 render_3d_concepts.py` for 1280x720 (about 4 min) or add `--preview` for a quick low-res pass. Pass scene names (`tide monolith salvage forest lighthouse`) to render only those.

| # | Game | Pitch | Core loop |
|---|------|-------|-----------|
| 1 | **Hollow Tide** | Pilot a skiff through a drowned megacity. | Sail, dive into flooded towers for salvage, upgrade the boat, outrun the rising water. |
| 2 | **Monolith** | A black slab appears in the desert and moves closer every night. | Walk, survive the heat and cold, decode the slab, choose to reach it or flee. |
| 3 | **Dead Orbit** | Zero-g salvage on a gutted space station. | Cut parts loose, tether them, manage oxygen and momentum, haul loot home. |
| 4 | **Signal Pine** | A dead radio tower in a foggy forest starts broadcasting. | Explore by lantern, tune the radio to find clues, avoid what hunts by sound. |
| 5 | **Last Light** | Keep the last lighthouse burning through an endless storm. | Gather fuel by day, steer the beam at night to guide ships and keep things out. |

For a real 3D game in C: **raylib** (easy, built-in 3D and shaders) or **SDL3 + OpenGL/Vulkan** (full control). Add **cglm** for math, **Jolt** or **ODE** for physics, and **miniaudio** for sound.
