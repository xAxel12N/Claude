#!/usr/bin/env bash
# Quantom.sh - build, train and control QuantumGPT on Linux.
#
#   ./Quantom.sh gui        the window: brain view, chat, drag-in .txt files, temperatures and fans
#   ./Quantom.sh            menu with start / stop buttons
#   ./Quantom.sh train      train here with live keys (SPACE start/stop, g sample, q quit)
#   ./Quantom.sh start      train in the background      ./Quantom.sh stop     save and stop it
#   ./Quantom.sh pause | resume | status | sample | gradcheck | gpucheck | build | clean
#   ./Quantom.sh train --gpu   train on the graphics card (the window has a Device: GPU switch)
#
# Extra options go straight to the program, e.g.
#   ./Quantom.sh start --opt momentum --sched step --anneal 0.5 --steps 6000
set -euo pipefail
cd "$(dirname "$0")"

BIN=build/quantum
DATA=${DATA:-data/input.txt}
CKPT=${CKPT:-quantum.ckpt}
PIDFILE=.quantum.pid
LOG=train.log
DATA_URL=https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

RAYLIB=third_party/raylib-5.5
GUI=build/quantum-gui

# raylib draws the window. The first build downloads it once (prebuilt on x86-64, else from source).
need_raylib() {
    [[ -f $RAYLIB/lib/libraylib.a ]] && return
    mkdir -p third_party
    echo "downloading raylib 5.5 (graphics library, one time) ..."
    if [[ $(uname -m) == x86_64 ]] &&
        curl -fsSL https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_linux_amd64.tar.gz | tar xz -C third_party; then
        mv third_party/raylib-5.5_linux_amd64 "$RAYLIB"
        return
    fi
    echo "building raylib from source ..."
    rm -rf third_party/raylib-src && mkdir -p third_party/raylib-src
    curl -fsSL https://github.com/raysan5/raylib/archive/refs/tags/5.5.tar.gz | tar xz -C third_party/raylib-src --strip-components=1
    if ! make -C third_party/raylib-src/src PLATFORM=PLATFORM_DESKTOP -j"$(nproc)" >/dev/null; then
        echo "raylib needs: sudo apt install build-essential libx11-dev libxrandr-dev libxi-dev libxcursor-dev libxinerama-dev libgl-dev" >&2
        exit 1
    fi
    mkdir -p "$RAYLIB/lib" "$RAYLIB/include"
    cp third_party/raylib-src/src/libraylib.a "$RAYLIB/lib/"
    cp third_party/raylib-src/src/raylib.h "$RAYLIB/include/"
}

# Links against libGL / libX11 even when only the runtime .so.1 / .so.6 is installed.
syslib() {
    local name=$1 so
    if ldconfig -p | grep -q " $name.so "; then echo "-l${name#lib}"; return; fi
    so=$(ldconfig -p | awk -v n="$name.so." '$1 ~ "^"n {print $NF; exit}')
    echo "${so:--l${name#lib}}"
}

build_gui() {
    need_raylib
    local cc=${CC:-cc} src=(gui.c sensors.c model.c optim.c data.c gpu.c) libs
    libs=("$RAYLIB/lib/libraylib.a" "$(syslib libGL)" "$(syslib libX11)" -lm -lpthread -ldl -lrt)
    mkdir -p build
    echo "building the window ..."
    if ! "$cc" -O3 -march=native -fcx-limited-range -fopenmp -std=gnu11 -I"$RAYLIB/include" -o "$GUI" "${src[@]}" "${libs[@]}" 2>build/gui-errors.txt &&
        ! "$cc" -O2 -std=gnu11 -I"$RAYLIB/include" -o "$GUI" "${src[@]}" "${libs[@]}" 2>build/gui-errors.txt; then
        cat build/gui-errors.txt >&2
        echo "if that is a missing library: sudo apt install build-essential libgl-dev libx11-dev" >&2
        exit 1
    fi
    echo "built $GUI"
}

gui() {
    local f fresh=0
    for f in gui.c sensors.c sensors.h model.c optim.c data.c gpu.c gpu.h qcore.h; do [[ $f -nt $GUI ]] && fresh=1; done
    [[ -x $GUI && $fresh == 0 ]] || build_gui
    mkdir -p texts
    if ! ls texts/*.txt texts/*.md >/dev/null 2>&1; then # start with Shakespeare so there is something to learn
        need_data
        cp "$DATA" texts/shakespeare.txt
    fi
    if p=$(pid); then echo "stopping the background run first (the window trains by itself)"; stop; fi
    "$GUI"
}

build() {
    local cc=${CC:-cc} src=(main.c model.c optim.c data.c gpu.c)
    mkdir -p build
    echo "building with $cc ..."
    if ! "$cc" -O3 -march=native -fcx-limited-range -fopenmp -std=gnu11 -o "$BIN" "${src[@]}" -lm -ldl 2>/dev/null; then
        echo "  (no OpenMP / native flags - building single-threaded)"
        "$cc" -O2 -std=gnu11 -o "$BIN" "${src[@]}" -lm -ldl
    fi
    echo "built $BIN"
}

need_build() {
    local f
    [[ -x $BIN ]] || { build; return; }
    for f in main.c model.c optim.c data.c gpu.c gpu.h qcore.h; do
        [[ $f -nt $BIN ]] && { build; return; }
    done
    return 0
}

need_data() {
    [[ -s $DATA ]] && return
    mkdir -p "$(dirname "$DATA")"
    echo "downloading Tiny Shakespeare (1 MB) ..."
    if command -v curl >/dev/null && curl -fsSL "$DATA_URL" -o "$DATA"; then return; fi
    if command -v wget >/dev/null && wget -q "$DATA_URL" -O "$DATA"; then return; fi
    rm -f "$DATA"
    echo "download failed - put any big plain-text file at $DATA (or set DATA=path)" >&2
    exit 1
}

pid() { [[ -f $PIDFILE ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && cat "$PIDFILE"; }

start() {
    need_build
    need_data
    if p=$(pid); then echo "already running (pid $p) - ./Quantom.sh stop"; return; fi
    nohup "$BIN" train --start --data "$DATA" --ckpt "$CKPT" "$@" </dev/null >"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    echo "started (pid $!) - log in $LOG, watch with: ./Quantom.sh status"
}

stop() {
    local p
    if ! p=$(pid); then echo "not running"; rm -f "$PIDFILE"; return; fi
    kill -TERM "$p"
    echo -n "stopping and saving "
    for _ in $(seq 1 120); do
        kill -0 "$p" 2>/dev/null || break
        echo -n "."
        sleep 0.5
    done
    echo
    rm -f "$PIDFILE"
    tail -n 1 "$LOG"
}

signal() { # pause / resume a background run
    local p
    if ! p=$(pid); then echo "not running"; return; fi
    kill "-$1" "$p"
    echo "$2"
}

status() {
    if p=$(pid); then echo "RUNNING (pid $p)"; else echo "not running"; fi
    [[ -f $LOG ]] && grep -a 'step' "$LOG" | tail -n 1
    [[ -f $CKPT ]] && echo "checkpoint: $CKPT ($(du -h "$CKPT" | cut -f1))"
    return 0
}

sample() {
    need_build
    [[ -f $CKPT ]] || { echo "no checkpoint yet - train first"; return 1; }
    "$BIN" sample --ckpt "$CKPT" "$@"
}

train() {
    need_build
    need_data
    if p=$(pid); then echo "a background run is active (pid $p) - stop it first"; return 1; fi
    "$BIN" train --data "$DATA" --ckpt "$CKPT" "$@"
}

menu() {
    while true; do
        echo
        echo "============ QuantumGPT ============"
        status | sed 's/^/  /'
        echo "------------------------------------"
        echo "  1) ▶  START   (train in background)"
        echo "  2) ■  STOP    (save and stop)"
        echo "  3) ⏸  PAUSE        4) ⏵  RESUME"
        echo "  5) ✎  Write text with the model"
        echo "  6) ⌨  Train here with live keys"
        echo "  7) ↻  Refresh      8) ✓  Gradient check"
        echo "  9) ✕  Quit menu (training keeps running)"
        echo "  G) ◉  Open the window (brain view + chat)"
        read -rp "choose: " c || exit 0
        case $c in
        1) start ;;
        2) stop ;;
        3) signal USR1 "paused" ;;
        4) signal USR2 "resumed" ;;
        5) read -rp "prompt: " pr; sample --prompt "$pr" --len 400 || true ;;
        6) train || true ;;
        7) ;;
        8) need_build; "$BIN" gradcheck || true ;;
        9 | q) exit 0 ;;
        g | G) gui ;;
        esac
    done
}

cmd=${1:-menu}
[[ $# -gt 0 ]] && shift
case $cmd in
build) build ;;
gui) gui ;;
data) need_data ;;
train) train "$@" ;;
start) start "$@" ;;
stop) stop ;;
pause) signal USR1 "paused" ;;
resume) signal USR2 "resumed" ;;
status) status ;;
sample) sample "$@" ;;
gradcheck) need_build; "$BIN" gradcheck ;;
gpucheck) need_build; "$BIN" gpucheck ;;
clean) rm -rf build "$CKPT" "$CKPT.best" "$LOG" "$PIDFILE" ;;
menu) menu ;;
help | -h | --help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//' ;;
*) echo "unknown command $cmd"; sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
