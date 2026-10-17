#!/usr/bin/env bash
# Quantom.sh - build, train and control QuantumGPT on Linux.
#
#   ./Quantom.sh            menu with start / stop buttons
#   ./Quantom.sh train      train here with live keys (SPACE start/stop, g sample, q quit)
#   ./Quantom.sh start      train in the background      ./Quantom.sh stop     save and stop it
#   ./Quantom.sh pause | resume | status | sample | gradcheck | build | clean
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

build() {
    local cc=${CC:-cc} src=(main.c model.c optim.c data.c)
    mkdir -p build
    echo "building with $cc ..."
    if ! "$cc" -O3 -march=native -fcx-limited-range -fopenmp -std=gnu11 -o "$BIN" "${src[@]}" -lm 2>/dev/null; then
        echo "  (no OpenMP / native flags - building single-threaded)"
        "$cc" -O2 -std=gnu11 -o "$BIN" "${src[@]}" -lm
    fi
    echo "built $BIN"
}

need_build() {
    local f
    [[ -x $BIN ]] || { build; return; }
    for f in main.c model.c optim.c data.c qcore.h; do
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
        esac
    done
}

cmd=${1:-menu}
[[ $# -gt 0 ]] && shift
case $cmd in
build) build ;;
data) need_data ;;
train) train "$@" ;;
start) start "$@" ;;
stop) stop ;;
pause) signal USR1 "paused" ;;
resume) signal USR2 "resumed" ;;
status) status ;;
sample) sample "$@" ;;
gradcheck) need_build; "$BIN" gradcheck ;;
clean) rm -rf build "$CKPT" "$CKPT.best" "$LOG" "$PIDFILE" ;;
menu) menu ;;
help | -h | --help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' ;;
*) echo "unknown command $cmd"; sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
