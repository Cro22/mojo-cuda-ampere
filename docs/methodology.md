# Methodology

## Goal

For each kernel, compare hand-written CUDA against hand-written Mojo *for the same
algorithm* on the same GPU, and place both against the hardware roofline. This is
not a "which language is faster" claim — it is "how much of the achievable
performance does each expression of the same idea capture."

## Shared CSV schema

Every program (CUDA binary or Mojo file) emits rows in one schema:

```
kernel,impl,variant,dtype,m,n,k,time_ms,gflops,gbytes_s,correct
```

| column | meaning |
|--------|---------|
| `kernel` | `reduction` \| `softmax` \| `matmul` |
| `impl` | `cuda` \| `mojo` |
| `variant` | algorithm variant (`naive`, `warp_shfl`, `online`, `tiled`, `regblock`, `cublas`) |
| `dtype` | element type (`f32`) |
| `m,n,k` | problem dims — reduction: `m`=elements; softmax: `m`=rows,`n`=cols; matmul: `M,N,K` |
| `time_ms` | **minimum** per-iteration time over the timed reps (steady state) |
| `gflops` | derived: work / time |
| `gbytes_s` | derived: DRAM traffic / time |
| `correct` | 1 if the result matches the reference within tolerance |

The harness (`bench/run.sh`) concatenates all rows into `results/results.csv` with
a single header.

## Timing

Both sides measure **device time only**, not host launch overhead, and report the
**minimum** over the timed iterations after a warmup (minimum, not mean, because
we want the achievable steady-state number with the least scheduler/DVFS noise).

- **CUDA** — `cudaEvent` pairs around each launch; warmup 3–10 iters, timed
  20–50 iters (`bench/bench.cuh::bench_time_ms`).
- **Mojo** — `DeviceContext.execution_time(closure, iters)`, which times `iters`
  back-to-back enqueues on the device timeline and returns total nanoseconds; we
  divide by `iters`. A separate warmup launch precedes it.

The two timers are not identical instruments, but both isolate device execution
of the kernel(s) under test, so the comparison is fair at the ~few-percent level
that matters here.

## Work and traffic models

| kernel | FLOPs | DRAM traffic (bytes) | bound by |
|--------|-------|----------------------|----------|
| reduction | `N-1` adds | `4N` (read once) | memory |
| softmax | `~5·M·N` (incl. `exp`) | `2·4·M·N` (read + write) | memory |
| matmul | `2·M·N·K` | `4·(MK+KN+MN)` (compute-bound; traffic shown for reference) | compute |

`gbytes_s` is the primary metric for reduction/softmax; `gflops` for matmul. The
minimal traffic model is used (each input read once) — the optimized kernels are
designed to hit exactly that, so the achieved bandwidth is a real efficiency
number, and for the naive variants the same denominator makes their lower
`gbytes_s` reflect wasted re-reads.

## Correctness

No kernel is timed without also being checked; `correct` must be 1 for a row to
count.

- **reduction** — GPU sum vs a host `Float64` accumulation, well-conditioned
  all-positive input, relative tolerance `1e-3`. (An input that sums to ~0 is a
  Float32-cancellation trap, not a kernel bug — inputs are chosen to avoid it.)
- **softmax** — full-matrix relative **L2** error vs a per-row host reference,
  tolerance `1e-4`.
- **matmul** — relative L2 error against the vendor oracle. CUDA uses cuBLAS as
  the reference; Mojo checks a deterministic 256-cell subset via host
  dot-products (a full 4096³ CPU matmul would take minutes), tolerance `1e-4`.

## Environment

| | |
|--|--|
| GPU | NVIDIA GeForce RTX 3090 (GA102, `sm_86`, 82 SMs) |
| CUDA | 13.3 (`nvcc`), `-O3 -std=c++20 -arch=sm_86` |
| Mojo / MAX | Mojo 1.0.0, `max==26.5.0`, via `uv` |
| OS | WSL2 Ubuntu on Windows 11, NVIDIA driver passthrough |

The SM count that sizes the reduction grid is **queried at runtime**
(`cudaDevAttrMultiProcessorCount` / `DeviceAttribute.MULTIPROCESSOR_COUNT`), so the
harness saturates whatever GPU it runs on rather than assuming the 3090.

## Reproducing

```bash
uv sync
./bench/run.sh            # -> results/results.csv (prints a table too)
uv run bench/plot.py      # -> results/*.png
sudo ./bench/profile.sh   # -> results/ncu/*  (optional; needs counter perms)
```
