#!/usr/bin/env bash
# Run the full CUDA-vs-Mojo shootout and collect one tidy CSV.
#
#   ./bench/run.sh                 # build + run everything -> results/results.csv
#   KERNELS="reduction softmax" ./bench/run.sh   # subset
#   LOCK_SM=1695 LOCK_MEM=9751 ./bench/run.sh     # override lock targets
#   NOLOCK=1 ./bench/run.sh        # skip the clock-lock attempt entirely
#
# Assumes: nvcc on PATH (or /usr/local/cuda/bin), and the uv venv with mojo+max
# already synced (`uv sync`). Mojo is invoked via .venv/bin/mojo.
#
# Clocks: GPU boost (DVFS) is the single biggest source of run-to-run variance on
# a 3090 (10-15%). We pin the SM and memory clocks before measuring so the
# median/IQR the kernels report reflects the kernel, not the governor. Locking
# needs elevated privileges; under WSL the GPU is owned by the *Windows* host
# driver, so run this once in an **Administrator** shell before ./run.sh:
#     nvidia-smi -lgc 1695,1695 && nvidia-smi -lmc 9751     (reset: -rgc / -rmc)
# If the lock is denied we carry on unlocked and say so in results/run-env.txt;
# the reported IQR then simply widens to include the DVFS spread.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/cuda/bin:$PATH:/usr/lib/wsl/lib"

MOJO="$ROOT/.venv/bin/mojo"
OUT="$ROOT/results/results.csv"
ENVF="$ROOT/results/run-env.txt"
KERNELS="${KERNELS:-reduction softmax matmul}"
LOCK_SM="${LOCK_SM:-1695}"
LOCK_MEM="${LOCK_MEM:-9751}"

mkdir -p "$ROOT/results"

# --- clock lock (best effort) ---------------------------------------------
# CLOCKS_PRELOCKED=1 tells the harness the clocks were already pinned from an
# elevated/host shell (the usual case under WSL, where this unprivileged user
# cannot lock but the Windows host can). We then just record it and don't touch
# the lock ourselves.
LOCKED="no"
if [ "${CLOCKS_PRELOCKED:-0}" = "1" ]; then
    LOCKED="yes (pre-locked externally)"
    echo ">> clocks assumed pre-locked (CLOCKS_PRELOCKED=1)"
elif [ "${NOLOCK:-0}" != "1" ]; then
    if nvidia-smi -lgc "${LOCK_SM},${LOCK_SM}" >/dev/null 2>&1 \
       && nvidia-smi -lmc "${LOCK_MEM}" >/dev/null 2>&1; then
        LOCKED="yes (sm=${LOCK_SM}MHz mem=${LOCK_MEM}MHz)"
        echo ">> clocks locked: sm=${LOCK_SM} mem=${LOCK_MEM}"
        # Release the lock when we're done, however we exit.
        trap 'nvidia-smi -rgc >/dev/null 2>&1 || true; nvidia-smi -rmc >/dev/null 2>&1 || true' EXIT
    else
        echo ">> WARNING: could not lock GPU clocks (need an elevated shell);"
        echo ">>          running with DVFS on — the reported IQR absorbs the spread."
    fi
fi

# --- background clock sampler: record the clocks the kernels actually ran at.
SAMPLES="$(mktemp)"
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits -lms 250 \
    > "$SAMPLES" 2>/dev/null &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" >/dev/null 2>&1 || true' EXIT

echo ">> building CUDA kernels (sm_86)"
make -C "$ROOT/bench" >/dev/null

# CSV header once (schema in docs/methodology.md).
echo "kernel,impl,variant,dtype,m,n,k,median_ms,p25_ms,p75_ms,n_runs,gflops,gbytes_s,correct" > "$OUT"

for k in $KERNELS; do
    echo ">> $k  [cuda]"
    "$ROOT/bench/build/$k" >> "$OUT"      # no --header: append rows only

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
SM_STATS="$(awk -F', ' 'NF>=2{print $1}' "$SAMPLES" | sort -n \
    | awk '{a[NR]=$1} END{if(NR)printf "min=%d median=%d max=%d (n=%d)",a[1],a[int(NR/2)+1],a[NR],NR}')"
rm -f "$SAMPLES"

{
    echo "# Benchmark run environment"
    echo "date_utc:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "gpu:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
    echo "driver:        $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)"
    echo "nvcc:          $(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1)"
    echo "clocks_locked: ${LOCKED}"
    echo "sm_clock_obs:  ${SM_STATS:-unavailable}   # MHz, sampled every 250ms during the run"
} > "$ENVF"

echo ">> wrote $OUT"
echo ">> wrote $ENVF"
cat "$ENVF"
echo
column -t -s, "$OUT" | sed 's/^/   /'
