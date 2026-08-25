#!/usr/bin/env python3
"""Vendor-reference softmax timing via torch (the `torch.softmax` yardstick).

Emits rows in the shared CSV schema with impl=torch,variant=vendor so the
softmax table has a vendor column just like matmul has cuBLAS:

    kernel,impl,variant,dtype,m,n,k,median_ms,p25_ms,p75_ms,n_runs,gflops,gbytes_s,correct

torch is an OPTIONAL dependency (a CUDA build is ~2.5 GB), deliberately kept out
of pyproject.toml. Install it into the venv only if you want this column:

    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124

If torch is absent this script prints nothing and exits 0, so run.sh can call it
unconditionally.
"""
import sys

try:
    import torch
except ModuleNotFoundError:
    sys.exit(0)

if not torch.cuda.is_available():
    sys.exit(0)

# Same shapes and work/traffic model as kernels/softmax.
SHAPES = [(16384, 1024), (4096, 4096), (1024, 16384)]
WARMUP, REPS = 5, 30


def pct(sorted_vals, q):
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    if lo + 1 >= len(sorted_vals):
        return sorted_vals[-1]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac


def bench(fn):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPS):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))  # ms
    samples.sort()
    return pct(samples, 0.5), pct(samples, 0.25), pct(samples, 0.75)


def main():
    dev = torch.device("cuda")
    for M, N in SHAPES:
        x = torch.rand(M, N, device=dev, dtype=torch.float32)
        med, p25, p75 = bench(lambda: torch.softmax(x, dim=1))
        total = M * N
        bytes_ = 2.0 * total * 4.0
        flops = 5.0 * total
        gflops = flops / (med * 1e6)
        gbytes = bytes_ / (med * 1e6)
        print(f"softmax,torch,vendor,f32,{M},{N},1,"
              f"{med:.6f},{p25:.6f},{p75:.6f},{REPS},{gflops:.3f},{gbytes:.3f},1")


if __name__ == "__main__":
    main()
