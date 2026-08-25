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
            for kf in ("m", "n", "k", "n_runs", "correct"):
                r[kf] = int(r[kf])
            for ff in ("median_ms", "p25_ms", "p75_ms", "gflops", "gbytes_s"):
                r[ff] = float(r[ff])
            rows.append(r)
    return rows


def iqr_err(row, metric):
    """Return (low, high) error-bar magnitudes for a throughput `metric`.

    Throughput is inversely proportional to time, so the fast quartile (p25 time)
    maps to the HIGH throughput bound and the slow quartile (p75) to the LOW one.
    Returns 0 when the row is absent (a missing series in the grouped bar)."""
    if not row:
        return 0.0, 0.0
    val = row[metric]
    hi = val * row["median_ms"] / row["p25_ms"]
    lo = val * row["median_ms"] / row["p75_ms"]
    # clamp tiny fp-noise negatives (p25==median==p75 -> exact 0 bounds)
    return max(0.0, val - lo), max(0.0, hi - val)


def grouped_bar(ax, labels, series, colors, ylabel, title, ceiling=None,
                ceiling_label=None, errs=None):
    import numpy as np
    x = np.arange(len(labels))
    n = len(series)
    w = 0.8 / n
    for i, (name, vals) in enumerate(series):
        yerr = None
        if errs is not None and errs[i] is not None:
            yerr = np.array(errs[i]).T  # shape (2, len) -> [lower row, upper row]
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=name, color=colors[i],
                      yerr=yerr, capsize=2.5,
                      error_kw=dict(elinewidth=0.8, ecolor="#333333"))
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
    r = [x for x in rows if x["kernel"] == "reduction" and x["variant"] in ("warp_shfl", "cub")]
    sizes = sorted({x["m"] for x in r})
    labels = [f"{s // (1 << 20)}M" for s in sizes]

    def rowset(impl, variant):
        picked = []
        for s in sizes:
            m = [x for x in r if x["impl"] == impl and x["variant"] == variant and x["m"] == s]
            picked.append(m[0] if m else None)
        return picked

    defs = [("CUDA warp_shfl", CUDA_C, rowset("cuda", "warp_shfl")),
            ("Mojo warp_shfl", MOJO_C, rowset("mojo", "warp_shfl")),
            ("cub (vendor)",   REF_C,  rowset("cuda", "cub"))]
    vals = [[(x["gbytes_s"] if x else 0.0) for x in rs] for _, _, rs in defs]
    errs = [[iqr_err(x, "gbytes_s") for x in rs] for _, _, rs in defs]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    grouped_bar(ax, labels, [(d[0], v) for d, v in zip(defs, vals)],
                [d[1] for d in defs], "GB/s",
                "Reduction (sum) — achieved DRAM bandwidth (median, IQR bars)",
                PEAK_BW, f"peak {PEAK_BW:.0f} GB/s", errs=errs)
    ax.set_xlabel("elements")
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "reduction_bandwidth.png")
    fig.savefig(out, dpi=130); print("wrote", out)


def plot_softmax(rows):
    r = [x for x in rows if x["kernel"] == "softmax"
         and x["variant"] in ("online", "vendor")]
    shapes = sorted({(x["m"], x["n"]) for x in r
                     if x["variant"] == "online"}, key=lambda s: s[1])
    labels = [f"{m}x{n}" for m, n in shapes]

    def rowset(impl):
        picked = []
        for s in shapes:
            m = [x for x in r if x["impl"] == impl and (x["m"], x["n"]) == s]
            picked.append(m[0] if m else None)
        return picked

    defs = [("CUDA online", CUDA_C, rowset("cuda")),
            ("Mojo online", MOJO_C, rowset("mojo"))]
    if any(x["impl"] == "torch" for x in r):   # only if the vendor ref was run
        defs.append(("torch (vendor)", REF_C, rowset("torch")))
    vals = [[(x["gbytes_s"] if x else 0.0) for x in rs] for _, _, rs in defs]
    errs = [[iqr_err(x, "gbytes_s") for x in rs] for _, _, rs in defs]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    grouped_bar(ax, labels, [(d[0], v) for d, v in zip(defs, vals)],
                [d[1] for d in defs], "GB/s",
                "Softmax (row-wise) — achieved DRAM bandwidth (median, IQR bars)",
                PEAK_BW, f"peak {PEAK_BW:.0f} GB/s", errs=errs)
    ax.set_xlabel("rows x cols")
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "softmax_bandwidth.png")
    fig.savefig(out, dpi=130); print("wrote", out)


def plot_matmul(rows):
    r = [x for x in rows if x["kernel"] == "matmul"]
    sizes = sorted({x["m"] for x in r})
    labels = [f"{s}³" for s in sizes]

    def rowset(impl, variant):
        picked = []
        for s in sizes:
            m = [x for x in r if x["impl"] == impl and x["variant"] == variant and x["m"] == s]
            picked.append(m[0] if m else None)
        return picked

    defs = [("CUDA tiled",    "#3a5a00", rowset("cuda", "tiled")),
            ("Mojo tiled",    "#b34700", rowset("mojo", "tiled")),
            ("CUDA regblock", CUDA_C,    rowset("cuda", "regblock")),
            ("Mojo regblock", MOJO_C,    rowset("mojo", "regblock")),
            ("cuBLAS",        "#1f77b4", rowset("cuda", "cublas"))]
    vals = [[(x["gflops"] if x else 0.0) for x in rs] for _, _, rs in defs]
    errs = [[iqr_err(x, "gflops") for x in rs] for _, _, rs in defs]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    grouped_bar(ax, labels, [(d[0], v) for d, v in zip(defs, vals)],
                [d[1] for d in defs], "GFLOP/s",
                "Matmul (SGEMM) — throughput (median, IQR bars)",
                PEAK_FP32, f"fp32 peak {PEAK_FP32/1000:.1f} TFLOP/s", errs=errs)
    ax.set_xlabel("M=N=K")
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "matmul_gflops.png")
    fig.savefig(out, dpi=130); print("wrote", out)


if __name__ == "__main__":
    rows = load()
    plot_reduction(rows)
    plot_softmax(rows)
    plot_matmul(rows)
