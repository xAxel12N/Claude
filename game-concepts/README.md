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
