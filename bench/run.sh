#!/usr/bin/env bash
# Run the full CUDA-vs-Mojo shootout and collect one tidy CSV.
#
#   ./bench/run.sh                 # build + run everything -> results/results.csv
#   KERNELS="reduction softmax" ./bench/run.sh   # subset
#
# Assumes: nvcc on PATH (or /usr/local/cuda/bin), and the uv venv with mojo+max
# already synced (`uv sync`). Mojo is invoked via .venv/bin/mojo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/cuda/bin:$PATH"

MOJO="$ROOT/.venv/bin/mojo"
OUT="$ROOT/results/results.csv"
KERNELS="${KERNELS:-reduction softmax matmul}"

mkdir -p "$ROOT/results"

echo ">> building CUDA kernels (sm_86)"
make -C "$ROOT/bench" >/dev/null

# CSV header once.
echo "kernel,impl,variant,dtype,m,n,k,time_ms,gflops,gbytes_s,correct" > "$OUT"

for k in $KERNELS; do
    echo ">> $k  [cuda]"
    "$ROOT/bench/build/$k" >> "$OUT"      # no --header: append rows only

    echo ">> $k  [mojo]"
    # Mojo prints warnings to stderr; keep only stdout CSV rows.
    "$MOJO" run "$ROOT/kernels/$k/mojo/$k.mojo" 2>/dev/null | grep "^$k," >> "$OUT"
done

echo ">> wrote $OUT"
column -t -s, "$OUT" | sed 's/^/   /'
