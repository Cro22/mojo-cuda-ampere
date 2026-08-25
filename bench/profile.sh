#!/usr/bin/env bash
# Capture Nsight Compute (ncu) profiles for the three optimized kernels.
#
#   ./bench/profile.sh          # writes results/ncu/*.txt and *.ncu-rep
#
# REQUIRES GPU performance-counter access. On a fresh WSL/driver install ncu
# reports ERR_NVGPUCTRPERM. Enable it ONE TIME with either:
#   * run this script under sudo:   sudo ./bench/profile.sh
#   * or lift the restriction globally (Linux host):
#       echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
#         | sudo tee /etc/modprobe.d/nvidia-profiler.conf && sudo reboot
# See https://developer.nvidia.com/ERR_NVGPUCTRPERM
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/usr/local/cuda/bin:$PATH"
OUT="$ROOT/results/ncu"; mkdir -p "$OUT"
make -C "$ROOT/bench" >/dev/null

SECTIONS="--section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy"

profile() {   # <name> <binary> <kernel-regex>
    local name="$1" bin="$2" kern="$3"
    echo ">> profiling $name ($kern)"
    # -s 24: skip to a steady-state launch; -c 1: one launch.
    ncu $SECTIONS -k "$kern" -s 24 -c 1 \
        --export "$OUT/$name" --force-overwrite "$bin" >/dev/null 2>&1 || true
    ncu --import "$OUT/$name.ncu-rep" --page details > "$OUT/$name.txt" 2>/dev/null || true
    echo "   -> $OUT/$name.txt"
}

profile reduction "$ROOT/bench/build/reduction" "reduce_warp_shfl"
profile softmax   "$ROOT/bench/build/softmax"   "softmax_online"
profile matmul    "$ROOT/bench/build/matmul"    "mm_regblock"

echo ">> done. Key metrics:"
grep -hE "Duration|Compute \(SM\)|Memory Throughput|DRAM Throughput|Achieved Occupancy" \
    "$OUT"/*.txt 2>/dev/null || echo "   (no reports — counters not accessible?)"
