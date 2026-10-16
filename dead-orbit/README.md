# Dead Orbit

A zero-g salvage horror-comedy for Linux, written in C with raylib. It's offline and single-player.

Cut parts off a wrecked space station, tether them, and haul them to your tug's green cargo bay before your oxygen runs out. Watch out for **Gloopers** (googly-eyed jelly blobs that hug the air out of you) and **Mimics** (crates with teeth). **Buddy**, your ship's AI, comments on everything.

All graphics are procedural shaders and all audio is synthesized at runtime, so the game needs no asset downloads. The only bundled files are the DejaVu fonts.

## Build

```sh
# Debian/Ubuntu
sudo apt install build-essential cmake git libx11-dev libxrandr-dev libxi-dev \
  libxcursor-dev libxinerama-dev libgl1-mesa-dev libasound2-dev
# Fedora: sudo dnf install gcc cmake git libX11-devel libXrandr-devel libXi-devel libXcursor-devel libXinerama-devel mesa-libGL-devel alsa-lib-devel
# Arch:   sudo pacman -S base-devel cmake git libx11 libxrandr libxi libxcursor libxinerama mesa alsa-lib

cmake -S . -B build
cmake --build build -j
./build/dead-orbit
```

The first configure downloads raylib 5.5 from GitHub, unless a system raylib 5.5 is installed. For a fully offline build, clone raylib once and pass `-DFETCHCONTENT_SOURCE_DIR_RAYLIB=/path/to/raylib`. `sudo cmake --install build` installs the game system-wide, including a desktop entry.

## Controls

| Key | Action |
|-----|--------|
| Mouse | Look |
| W A S D | Thrust |
| Space / Ctrl | Up / down |
| Shift | Boost (mash it to shake off a Glooper) |
| X | Brake |
| Left mouse | Plasma torch: cut bolts, pop Gloopers, kill Mimics |
| Right mouse / F | Fire a tether at loose salvage (3 max) |
| R (hold) | Reel tethers in |
| E | Release all tethers |
| Esc | Pause |

Fly into the green bay to deposit loot and refill O2 from the ship's limited reserve. Blue canisters give +30 O2. Reach the quota, then return to the bay to win. Difficulty (Easy / Normal / Nightmare) changes the quota, air supply and how many monsters show up.

Settings and best scores are saved in `~/.local/share/dead-orbit/settings.cfg`.

## Code

- `src/main.c`: window, state machine, input, `--autotest DIR` (scripted run that saves screenshots)
- `src/world.c`: station, player physics, tethers, torch, creatures, ship, Buddy, HUD
- `src/gfx.c`: lit procedural materials, sky, planet, particles, bloom and post effects
- `src/audio.c`: synthesized soundtrack and sound effects
- `src/ui.c`: menus and screens
- `src/settings.c`: config and save file
