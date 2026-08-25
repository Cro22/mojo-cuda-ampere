# mojo-cuda-ampere

A head-to-head **CUDA vs Mojo** GPU kernel shootout on **NVIDIA Ampere** (RTX 3090,
`sm_86`). Three kernels (reduction, softmax, matmul), each written twice, optimized
as far as is reasonable by hand, timed with the same methodology, and plotted
against the hardware roofline.

The question this answers: *how close does hand-written Mojo get to hand-written
CUDA (and to the vendor libraries, cuBLAS / CUB / torch) on the same GPU, for the
same algorithm?*

## Results (RTX 3090, CUDA 13.3, Mojo 1.0.0 / MAX 26.5.0)

Clocks **pinned at SM 1695 MHz** for the run; each number is the **median of 30
timed samples** and `± ` is the 25/75 inter-quartile range (IQR). "Indistinguishable"
means the CUDA and Mojo IQRs overlap, so the difference is inside the measurement
noise; where they do not overlap the gap is real and quoted. Raw rows and the
`run-env.txt` clock record are in `results/`.

| Kernel | Metric | Best CUDA | Best Mojo | Mojo / CUDA | Vendor ref | Verdict |
|--------|--------|----------:|----------:|:-----------:|-----------:|---------|
| reduction (256M f32)   | GB/s    | **891** (warp_shfl) | 889 (warp_shfl) | 99.7% | 888 (CUB)     | **indistinguishable** |
| softmax (16384×1024)   | GB/s    | **730** (online)    | 713 (online)    | 97.6% | torch (opt.)  | **indistinguishable** |
| softmax (1024×16384)   | GB/s    | **535** (online)    | 498 (online)    | 93.2% | torch (opt.)  | CUDA +7% |
| matmul (1024³ f32)     | GFLOP/s | **9 279** (regblock)| 8 491 (regblock)| 91.5% | cuBLAS 18 079 | CUDA +9% |
| matmul (4096³ f32)     | GFLOP/s | **16 442** (regblock)| 13 099 (regblock)| 79.7% | cuBLAS 22 897 | CUDA +25% |

The picture splits cleanly by what limits each kernel:

- **Memory-bound** (reduction, softmax): both languages sit near the 936 GB/s
  GDDR6X ceiling and are **statistically indistinguishable** at the sizes that
  saturate the bus. The hand-written Mojo reduction matches both the hand-written
  CUDA kernel *and* `cub::DeviceReduce` to within ~3 GB/s at 256M (all ~95% of the
  bus). Softmax is a dead heat on square/wide-block shapes; only the very wide
  1024×16384 shape opens a real ~7% gap, where the per-block reduction overhead is
  a larger share of a short kernel.
- **Compute-bound** (matmul): once both sides stage A/B through `float4`-vectorized
  loads with an identical 8×8 register-tile inner loop, Mojo reaches **92% of the
  CUDA kernel at 1024³** and settles at **~80% at 2048³/4096³**. The residual gap is
  **double buffering** (overlapping the next tile's global loads with the current
  tile's FMAs) and instruction scheduling, which cuBLAS and the tuned CUDA kernel
  still do better. It is not memory movement and not the arithmetic.

> On methodology: the previous version of this table reported single-run figures
> and, worse, compared a CUDA *minimum* against a Mojo *mean*. Both sides now report
> the **median over 30 locked-clock samples with the IQR**, which is why some gaps
> moved: the old "95% / 87%" memory-bound gaps were measurement artifacts (they are
> now dead heats), and the old "84% at 4096³" matmul was an optimistic single run
> (the honest locked-clock median is ~80%).

![matmul](results/matmul_gflops.png)
![reduction](results/reduction_bandwidth.png)
![softmax](results/softmax_bandwidth.png)

## What it costs (why this repo exists)

Benchmarks are only interesting if they translate into a bill. Take the compute-bound
case, the only one where the languages actually differ, and price it. A 3090 rents
for roughly **$0.22/hr** on community clouds (mid-2026 spot; the numbers below scale
linearly with whatever price you pay).

Cost to push **1B tokens through one 4096-wide `float32` projection** (a single
`(1e9 × 4096) · (4096 × 4096)` SGEMM = 33.6 PFLOP), at the measured 4096³ rates:

| impl | GFLOP/s | wall time | cost / 1B tokens | vs CUDA |
|------|--------:|----------:|-----------------:|--------:|
| cuBLAS        | 22 897 | 24.5 min | **$0.090** | 0.72× |
| CUDA regblock | 16 442 | 34.1 min | **$0.125** | 1.00× |
| Mojo regblock | 13 099 | 42.8 min | **$0.157** | 1.25× |

So on this fp32 kernel the Mojo implementation costs **~25% more per token than the
hand-written CUDA one, and ~75% more than cuBLAS**, purely from the throughput gap.
On the **memory-bound** kernels the implementations are within measurement noise, so
they cost the same to run.

Caveats, stated plainly: this is a single fp32 GEMM with no tensor cores, so it is a
*language-vs-language* cost of the same kernel, not a production inference cost (real
serving uses fp16/bf16 tensor cores and is an order of magnitude cheaper per token).
The point is the **shape** of the answer: for bandwidth-bound work Mojo is already
free of penalty; for compute-bound work the ~20% throughput gap is also a ~25% cost
gap, and closing it (double buffering) is worth real money at scale.

## Layout

```
kernels/
  reduction/  { cuda/reduction.cu  mojo/reduction.mojo  README.md }
  softmax/    { cuda/softmax.cu    mojo/softmax.mojo    README.md }
  matmul/     { cuda/matmul.cu     mojo/matmul.mojo     README.md }
bench/        Makefile, run.sh (harness -> CSV), plot.py, profile.sh (ncu),
              vendor_softmax.py (optional torch ref)
results/      results.csv (raw), run-env.txt (clocks), *.png (charts), ncu/ (profiles)
docs/         methodology.md, roofline-3090.md
```

Every kernel program is self-contained: it runs its own size sweep, checks
correctness against a reference, and prints rows in one shared CSV schema
(`kernel,impl,variant,dtype,m,n,k,median_ms,p25_ms,p75_ms,n_runs,gflops,gbytes_s,correct`).
See [docs/methodology.md](docs/methodology.md).

## Setup

Mojo runs on Linux; this repo is developed under WSL2 (Ubuntu) with the NVIDIA
driver exposing the 3090. You need **CUDA** (`nvcc`) and a **uv** environment with
**Mojo + MAX**.

```bash
# 1. Python env with Mojo + MAX (the max package ships the GPU host runtime and
#    the layout library; the bare mojo wheel alone cannot launch kernels).
uv sync

# 2. (recommended) pin GPU clocks so the medians are not chasing the boost
#    governor. Under WSL the GPU is owned by the Windows host, so run this once in
#    an Administrator shell (reset afterwards with -rgc / -rmc):
#      nvidia-smi -lgc 1695,1695 && nvidia-smi -lmc 9751

# 3. Build + run everything -> results/results.csv + results/run-env.txt
./bench/run.sh

# 4. Charts
uv run bench/plot.py

# 5. (optional) the torch vendor reference for softmax (~2.5 GB CUDA build)
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124

# 6. (optional) Nsight Compute profiles (needs GPU counter permissions)
sudo ./bench/profile.sh
```

Run a single side by hand:

```bash
nvcc -O3 -std=c++20 -arch=sm_86 -lcublas -o /tmp/mm kernels/matmul/cuda/matmul.cu && /tmp/mm --header
.venv/bin/mojo run kernels/matmul/mojo/matmul.mojo
```

## What "optimized" means here

- **reduction**: grid-stride `float4`-vectorized load, warp-shuffle (`__shfl_down` /
  `warp.sum`) block reduction, one atomic/partial per block. Memory-bound; targets
  peak bandwidth. Referenced against `cub::DeviceReduce::Sum`.
- **softmax**: one block per row, single-pass **online softmax** (fused max+sum) with
  a warp-shuffle reduction over `(max, sum)` pairs, vs a naive three-pass baseline.
  Referenced against `torch.softmax` (optional).
- **matmul**: 128×128 block tile, `BK=8`, an **8×8 register tile per thread** (256
  threads/block), shared-memory staging, `float4` loads. Compared against a simple
  32×32 tiled kernel and against **cuBLAS**.

Details and the per-kernel optimization story are in each `kernels/*/README.md`.
