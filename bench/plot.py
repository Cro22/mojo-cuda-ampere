#!/usr/bin/env python3
"""Render CUDA-vs-Mojo comparison charts from results/results.csv.

    uv run bench/plot.py        # writes results/*.png

Each chart overlays the RTX 3090 hardware ceiling (measured/spec) so the bars
are read against the roofline, not in isolation.
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "results.csv")

# RTX 3090 (GA102) ceilings.
PEAK_BW = 936.2      # GB/s, 384-bit GDDR6X @ 19.5 Gbps
PEAK_FP32 = 35580.0  # GFLOP/s, 10496 CUDA cores @ ~1.695 GHz x 2

CUDA_C = "#76b900"   # NVIDIA green
MOJO_C = "#ff5f1f"   # Mojo orange
REF_C = "#888888"


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            for kf in ("m", "n", "k", "correct"):
                r[kf] = int(r[kf])
            for ff in ("time_ms", "gflops", "gbytes_s"):
                r[ff] = float(r[ff])
            rows.append(r)
    return rows


def grouped_bar(ax, labels, series, colors, ylabel, title, ceiling=None, ceiling_label=None):
    import numpy as np
    x = np.arange(len(labels))
    n = len(series)
    w = 0.8 / n
    for i, (name, vals) in enumerate(series):
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=name, color=colors[i])
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=7)
    if ceiling:
        ax.axhline(ceiling, ls="--", lw=1, color=REF_C)
        ax.text(len(labels) - 0.5, ceiling, f"  {ceiling_label}", color=REF_C,
                va="bottom", ha="right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_reduction(rows):
    r = [x for x in rows if x["kernel"] == "reduction" and x["variant"] == "warp_shfl"]
    sizes = sorted({x["m"] for x in r})
    labels = [f"{s // (1 << 20)}M" for s in sizes]
    cuda = [next(x["gbytes_s"] for x in r if x["impl"] == "cuda" and x["m"] == s) for s in sizes]
    mojo = [next(x["gbytes_s"] for x in r if x["impl"] == "mojo" and x["m"] == s) for s in sizes]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    grouped_bar(ax, labels, [("CUDA warp_shfl", cuda), ("Mojo warp_shfl", mojo)],
                [CUDA_C, MOJO_C], "GB/s", "Reduction (sum) — achieved DRAM bandwidth",
                PEAK_BW, f"peak {PEAK_BW:.0f} GB/s")
    ax.set_xlabel("elements")
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "reduction_bandwidth.png")
    fig.savefig(out, dpi=130); print("wrote", out)


def plot_softmax(rows):
    r = [x for x in rows if x["kernel"] == "softmax" and x["variant"] == "online"]
    shapes = sorted({(x["m"], x["n"]) for x in r}, key=lambda s: s[1])
    labels = [f"{m}x{n}" for m, n in shapes]
    cuda = [next(x["gbytes_s"] for x in r if x["impl"] == "cuda" and (x["m"], x["n"]) == s) for s in shapes]
    mojo = [next(x["gbytes_s"] for x in r if x["impl"] == "mojo" and (x["m"], x["n"]) == s) for s in shapes]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    grouped_bar(ax, labels, [("CUDA online", cuda), ("Mojo online", mojo)],
                [CUDA_C, MOJO_C], "GB/s", "Softmax (row-wise) — achieved DRAM bandwidth",
                PEAK_BW, f"peak {PEAK_BW:.0f} GB/s")
    ax.set_xlabel("rows x cols")
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "softmax_bandwidth.png")
    fig.savefig(out, dpi=130); print("wrote", out)


def plot_matmul(rows):
    r = [x for x in rows if x["kernel"] == "matmul"]
    sizes = sorted({x["m"] for x in r})
    labels = [f"{s}³" for s in sizes]

    def series(impl, variant):
        out = []
        for s in sizes:
            m = [x for x in r if x["impl"] == impl and x["variant"] == variant and x["m"] == s]
            out.append(m[0]["gflops"] if m else 0.0)
        return out

    fig, ax = plt.subplots(figsize=(8, 4.6))
    grouped_bar(ax, labels, [
        ("CUDA tiled", series("cuda", "tiled")),
        ("Mojo tiled", series("mojo", "tiled")),
        ("CUDA regblock", series("cuda", "regblock")),
        ("Mojo regblock", series("mojo", "regblock")),
        ("cuBLAS", series("cuda", "cublas")),
    ], ["#3a5a00", "#b34700", CUDA_C, MOJO_C, "#1f77b4"],
        "GFLOP/s", "Matmul (SGEMM) — throughput",
        PEAK_FP32, f"fp32 peak {PEAK_FP32/1000:.1f} TFLOP/s")
    ax.set_xlabel("M=N=K")
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "matmul_gflops.png")
    fig.savefig(out, dpi=130); print("wrote", out)


if __name__ == "__main__":
    rows = load()
    plot_reduction(rows)
    plot_softmax(rows)
    plot_matmul(rows)
