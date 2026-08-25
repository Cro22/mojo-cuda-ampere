# Methodology

## Goal

For each kernel, compare hand-written CUDA against hand-written Mojo *for the same
algorithm* on the same GPU, and place both against the hardware roofline. This is
not a "which language is faster" claim, it is "how much of the achievable
performance does each expression of the same idea capture."

## Shared CSV schema

Every program (CUDA binary or Mojo file) emits rows in one schema:

```
kernel,impl,variant,dtype,m,n,k,median_ms,p25_ms,p75_ms,n_runs,gflops,gbytes_s,correct
```

| column | meaning |
|--------|---------|
| `kernel` | `reduction` \| `softmax` \| `matmul` |
| `impl` | `cuda` \| `mojo` \| `torch` (vendor ref) |
| `variant` | algorithm variant (`naive`, `warp_shfl`, `cub`, `online`, `vendor`, `tiled`, `regblock`, `cublas`) |
| `dtype` | element type (`f32`) |
| `m,n,k` | problem dims, reduction: `m`=elements; softmax: `m`=rows,`n`=cols; matmul: `M,N,K` |
| `median_ms` | **median** per-iteration time over `n_runs` timed samples (steady state) |
| `p25_ms`, `p75_ms` | 25th / 75th percentile of the same samples, the inter-quartile range |
| `n_runs` | number of timed samples the quartiles are computed from |
| `gflops` | derived: work / `median_ms` |
| `gbytes_s` | derived: DRAM traffic / `median_ms` |
| `correct` | 1 if the result matches the reference within tolerance |

The harness (`bench/run.sh`) concatenates all rows into `results/results.csv` with
a single header, and writes the run's clock/driver state to `results/run-env.txt`.

## Timing

Both sides measure **device time only**, not host launch overhead, and both
report the **same statistic**: the **median** of `n_runs = 30` timed samples
(after 5 untimed warmup iterations), together with the **25th/75th percentiles**
(the inter-quartile range, IQR).

Why median + IQR rather than a single minimum:

- A single number, whether a lucky minimum or a one-shot run, cannot be told
  apart from measurement noise. On a 3090 the boost governor alone moves timings
  10–15% between runs. Reporting the IQR makes that spread **visible** instead of
  hiding it, and lets the write-up say "indistinguishable" only when the CUDA and
  Mojo quartile ranges actually overlap, and "X% slower" only when they don't.
- The **median** is a robust central estimate (unlike the mean it ignores the
  occasional descheduling spike; unlike the minimum it is not a single best-case
  sample). Both sides use it, so the comparison is apples-to-apples, the earlier
  harness mixed a CUDA *minimum* with a Mojo *mean*, which is not.

Implementations:

- **CUDA**, `cudaEvent` pairs around each launch collect `n_runs` per-iteration
  samples; `bench/bench.cuh::bench_time` sorts them and returns median / p25 / p75.
- **Mojo**, `DeviceContext.execution_time(closure, 1)` times one iteration on the
  device timeline; we call it `n_runs` times and take the same three percentiles.
- **torch** (softmax vendor ref), `torch.cuda.Event` timing, identical protocol.

### Clocks (DVFS)

GPU boost is the dominant source of run-to-run variance. `bench/run.sh` pins the
SM and memory clocks (`nvidia-smi -lgc/-lmc`) before measuring so the reported
distribution reflects the kernel, not the governor. Locking needs elevated
privileges; under WSL the GPU is owned by the Windows host driver, so the lock is
issued from an Administrator shell (see `run.sh` header). If the lock is denied
the harness records `clocks_locked: no` in `results/run-env.txt` and continues -
the reported IQR then simply widens to include the DVFS spread rather than hiding
it. The actual SM clock is sampled every 250 ms during the run and its
min/median/max are recorded in the same file.

## Vendor references

Each kernel is also measured against the tuned NVIDIA/vendor implementation so the
hand-written numbers are read against "how close to the library", not in a vacuum:

| kernel | vendor reference | variant |
|--------|------------------|---------|
| reduction | `cub::DeviceReduce::Sum` (CUB, ships with CUDA) | `cub` |
| softmax | `torch.softmax` (optional; see `bench/vendor_softmax.py`) | `vendor` |
| matmul | cuBLAS `cublasSgemm` | `cublas` |

## Work and traffic models

| kernel | FLOPs | DRAM traffic (bytes) | bound by |
|--------|-------|----------------------|----------|
| reduction | `N-1` adds | `4N` (read once) | memory |
| softmax | `~5·M·N` (incl. `exp`) | `2·4·M·N` (read + write) | memory |
| matmul | `2·M·N·K` | `4·(MK+KN+MN)` (compute-bound; traffic shown for reference) | compute |

`gbytes_s` is the primary metric for reduction/softmax; `gflops` for matmul. The
minimal traffic model is used (each input read once), the optimized kernels are
designed to hit exactly that, so the achieved bandwidth is a real efficiency
number, and for the naive variants the same denominator makes their lower
`gbytes_s` reflect wasted re-reads.

## Correctness

No kernel is timed without also being checked; `correct` must be 1 for a row to
count.

- **reduction**, GPU sum vs a host `Float64` accumulation, well-conditioned
  all-positive input, relative tolerance `1e-3`. (An input that sums to ~0 is a
  Float32-cancellation trap, not a kernel bug, inputs are chosen to avoid it.)
- **softmax**, full-matrix relative **L2** error vs a per-row host reference,
  tolerance `1e-4`.
- **matmul**, relative L2 error against the vendor oracle. CUDA uses cuBLAS as
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
# optional: the torch vendor ref for softmax (~2.5 GB CUDA build; not a hard dep)
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
# optional: pin clocks first, from an Administrator shell (WSL: on the Windows host)
#   nvidia-smi -lgc 1695,1695 && nvidia-smi -lmc 9751     (reset: -rgc / -rmc)

./bench/run.sh            # -> results/results.csv + results/run-env.txt (prints a table)
uv run bench/plot.py      # -> results/*.png
sudo ./bench/profile.sh   # -> results/ncu/*  (optional; needs counter perms)
```
