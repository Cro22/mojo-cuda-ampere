#!/usr/bin/env bash
# Run the full shootout and collect one tidy CSV. Works on NVIDIA (CUDA) or AMD
# (ROCm/HIP): the platform is auto-detected from which GPU stack is present, and
# the clock-lock / clock-sampling / env-record blocks branch accordingly. The
# Mojo side is identical on both — one source, whichever accelerator MAX finds.
#
#   ./bench/run.sh                 # build + run everything -> results/results.csv
#   KERNELS="reduction softmax" ./bench/run.sh   # subset
#   LOCK_SM=1695 LOCK_MEM=9751 ./bench/run.sh     # NVIDIA: override lock targets
#   NOLOCK=1 ./bench/run.sh        # skip the clock-lock attempt entirely
#
# Assumes: the uv venv with mojo+max synced (`uv sync`), and either nvcc (NVIDIA)
# or hipcc+rocminfo (AMD) on PATH. Mojo is invoked via .venv/bin/mojo.
#
# Clocks: GPU boost (DVFS) is the single biggest source of run-to-run variance on
# a 3090 (10-15%). We pin the clocks before measuring so the median/IQR reflects
# the kernel, not the governor. Locking needs elevated privileges; under WSL the
# GPU is owned by the *Windows* host driver, and on a shared cloud box you may
# lack the privilege at all. If the lock is denied we carry on unlocked and say
# so in results/run-env.txt; the reported IQR then widens to absorb the spread.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/cuda/bin:$PATH:/usr/lib/wsl/lib:/opt/rocm/bin"

MOJO="$ROOT/.venv/bin/mojo"
OUT="$ROOT/results/results.csv"
ENVF="$ROOT/results/run-env.txt"
KERNELS="${KERNELS:-reduction softmax matmul}"
LOCK_SM="${LOCK_SM:-1695}"
LOCK_MEM="${LOCK_MEM:-9751}"

mkdir -p "$ROOT/results"

# --- platform detection ----------------------------------------------------
# rocminfo naming a gfx target => AMD/ROCm; otherwise assume NVIDIA/CUDA. This is
# the same signal the Makefile branches on, so build and harness always agree.
GFX="$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)"
if [ -n "$GFX" ]; then PLATFORM="rocm"; else PLATFORM="cuda"; fi
echo ">> platform: $PLATFORM${GFX:+ ($GFX)}"

# --- clock lock (best effort) ---------------------------------------------
# CLOCKS_PRELOCKED=1 tells the harness the clocks were already pinned from an
# elevated/host shell (the usual case under WSL). We then just record it.
LOCKED="no"
if [ "${CLOCKS_PRELOCKED:-0}" = "1" ]; then
    LOCKED="yes (pre-locked externally)"
    echo ">> clocks assumed pre-locked (CLOCKS_PRELOCKED=1)"
elif [ "${NOLOCK:-0}" != "1" ]; then
    if [ "$PLATFORM" = "cuda" ]; then
        if nvidia-smi -lgc "${LOCK_SM},${LOCK_SM}" >/dev/null 2>&1 \
           && nvidia-smi -lmc "${LOCK_MEM}" >/dev/null 2>&1; then
            LOCKED="yes (sm=${LOCK_SM}MHz mem=${LOCK_MEM}MHz)"
            echo ">> clocks locked: sm=${LOCK_SM} mem=${LOCK_MEM}"
            trap 'nvidia-smi -rgc >/dev/null 2>&1 || true; nvidia-smi -rmc >/dev/null 2>&1 || true' EXIT
        else
            echo ">> WARNING: could not lock GPU clocks (need an elevated shell);"
            echo ">>          running with DVFS on — the reported IQR absorbs the spread."
        fi
    else
        # ROCm: there is no per-MHz -lgc equivalent for a plain user; the honest
        # analogue is forcing the top DVFS state. Needs privilege on most boxes.
        if rocm-smi --setperflevel high >/dev/null 2>&1; then
            LOCKED="yes (perf_level=high)"
            echo ">> clocks pinned: rocm-smi --setperflevel high"
            trap 'rocm-smi --setperflevel auto >/dev/null 2>&1 || true' EXIT
        else
            echo ">> WARNING: could not pin GPU clocks (rocm-smi needs privilege);"
            echo ">>          running with DVFS on — the reported IQR absorbs the spread."
        fi
    fi
fi

# --- background clock sampler: record the clocks the kernels actually ran at.
SAMPLES="$(mktemp)"
if [ "$PLATFORM" = "cuda" ]; then
    nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits -lms 250 \
        > "$SAMPLES" 2>/dev/null &
    SAMPLER_PID=$!
else
    # rocm-smi has no built-in poll interval; loop and scrape the SCLK MHz.
    ( while true; do
        rocm-smi --showsclkclk 2>/dev/null | grep -oE '[0-9]+Mhz' | head -1 | tr -d 'Mhz'
        sleep 0.25
      done ) > "$SAMPLES" 2>/dev/null &
    SAMPLER_PID=$!
fi
trap 'kill "$SAMPLER_PID" >/dev/null 2>&1 || true' EXIT

echo ">> building C++ kernels ($PLATFORM${GFX:+ $GFX})"
make -C "$ROOT/bench" >/dev/null

# CSV header once (schema in docs/methodology.md).
echo "kernel,impl,variant,dtype,m,n,k,median_ms,p25_ms,p75_ms,n_runs,gflops,gbytes_s,correct" > "$OUT"

for k in $KERNELS; do
    CPP_IMPL="$([ "$PLATFORM" = "cuda" ] && echo cuda || echo hip)"
    echo ">> $k  [$CPP_IMPL]"
    "$ROOT/bench/build/$k" >> "$OUT"      # binary emits its own rows (impl=cuda|hip)

    echo ">> $k  [mojo]"
    # Mojo prints warnings to stderr; keep only stdout CSV rows.
    "$MOJO" run "$ROOT/kernels/$k/mojo/$k.mojo" 2>/dev/null | grep "^$k," >> "$OUT"

    # Optional vendor reference for softmax (torch). No-ops if torch is absent.
    if [ "$k" = "softmax" ]; then
        echo ">> $k  [torch vendor ref]"
        "$ROOT/.venv/bin/python" "$ROOT/bench/vendor_softmax.py" | grep "^softmax," >> "$OUT" || true
    fi
done

# --- summarize the sampled clocks and write the env record -----------------
kill "$SAMPLER_PID" >/dev/null 2>&1 || true
if [ "$PLATFORM" = "cuda" ]; then
    SM_STATS="$(awk -F', ' 'NF>=2{print $1}' "$SAMPLES" | sort -n \
        | awk '{a[NR]=$1} END{if(NR)printf "min=%d median=%d max=%d (n=%d)",a[1],a[int(NR/2)+1],a[NR],NR}')"
else
    SM_STATS="$(grep -E '^[0-9]+$' "$SAMPLES" | sort -n \
        | awk '{a[NR]=$1} END{if(NR)printf "min=%d median=%d max=%d (n=%d)",a[1],a[int(NR/2)+1],a[NR],NR}')"
fi
rm -f "$SAMPLES"

{
    echo "# Benchmark run environment"
    echo "date_utc:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "platform:      $PLATFORM${GFX:+ ($GFX)}"
    if [ "$PLATFORM" = "cuda" ]; then
        echo "gpu:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
        echo "driver:        $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)"
        echo "compiler:      $(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1)"
    else
        echo "gpu:           $(rocm-smi --showproductname 2>/dev/null | grep -m1 -i 'card series\|product name' | cut -d: -f2- | xargs)"
        echo "driver:        $(cat /sys/module/amdgpu/version 2>/dev/null || echo unknown)"
        echo "compiler:      $(hipcc --version 2>/dev/null | grep -m1 -oE 'HIP version[: ]+[0-9.]+')"
    fi
    echo "clocks_locked: ${LOCKED}"
    echo "sm_clock_obs:  ${SM_STATS:-unavailable}   # MHz, sampled every 250ms during the run"
} > "$ENVF"

echo ">> wrote $OUT"
echo ">> wrote $ENVF"
cat "$ENVF"
echo
column -t -s, "$OUT" | sed 's/^/   /'
