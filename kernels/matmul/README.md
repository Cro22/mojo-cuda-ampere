# Matmul (SGEMM)

`C = A × B`, `(M×K)·(K×N)`, row-major `float32`. **Compute-bound** once tiled, the
metric is GFLOP/s against the 35.6 TFLOP/s fp32 peak. `2·M·N·K` flops.

## Variants

| variant | tile | per-thread | idea |
|---------|------|-----------|------|
| `naive` | n/a | 1 output | one thread per `C` element, all reads from global memory. |
| `tiled` | 32×32 | 1 output | shared-memory tiles; each element reused 32×. |
| `regblock` | 128×128, `BK=8` | **8×8 register tile** | 256 threads/block, each computing an 8×8 block of `C` from register-resident operands. The optimized kernel. |
| `cublas` | n/a | n/a | vendor SGEMM, the reference / correctness oracle (CUDA only). |

### regblock structure (both languages)

- Block computes a 128×128 tile of `C`. 256 threads, each owning an 8×8 sub-tile.
- Inner `k`-loop over `BK=8`: load an 8-vector of `A` and `B` operands into
  registers (`regM`, `regN`), then do 64 FMAs into the register accumulator.
- Both languages stage `As` **transposed** and load A/B through `float4`
  (CUDA: `reinterpret_cast<float4*>`, Mojo: `A.ptr.unsafe_load[width=4]`). Since
  `BM·BK/4 = BK·BN/4 = 256 = threads/block`, each thread issues exactly one
  coalesced float4 load per tile. The compute inner loop is structurally identical.

## CUDA ↔ Mojo mapping

| CUDA | Mojo |
|------|------|
| `__shared__ float As[BK][BM]` | `stack_allocation[...SHARED](row_major[BK, BM]())` |
| register arrays `float results[64]` | `stack_allocation[dtype](row_major[TM,TN]()).fill(0)` (local) |
| `#pragma unroll` | `comptime for` (compile-time unroll) |
| `reinterpret_cast<float4*>` loads | `A.ptr.unsafe_load[width=4](offset)` |

## Results (GFLOP/s, clocks locked @1695 MHz, median of 30)

| size | CUDA regblock | Mojo regblock | Mojo/CUDA | cuBLAS |
|------|-------------:|-------------:|:---------:|-------:|
| 1024³ | 9 279 | 8 497 | **91.6%** | 18 118 |
| 2048³ | 14 526 | 11 507 | 79.2% | 22 520 |
| 4096³ | **17 370** | **13 780** | 79.3% | 23 385 |

Register blocking is a ~7x jump over the simple 32×32 tiled kernel (~2 400 GFLOP/s).
With `float4`-vectorized loads on **both** sides, the Mojo/CUDA ratio is **92% at
1024³** and settles at **~80% at 2048³ and 4096³**. The residual gap is **double
buffering** (overlapping the next tile's global loads with the current tile's FMAs)
and instruction scheduling, not memory movement and not the arithmetic (the 8×8
register inner loop is identical).

The ratio dropping from 92% to ~80% as the problem grows is consistent with the
double-buffering explanation: a larger `K` means more `BK`-tiles in the main loop, so
more iterations where CUDA's overlap of the next tile's loads with the current tile's
FMAs hides latency that the Mojo kernel exposes. At 1024³ there are only 128 such
iterations and the vectorized loads keep Mojo close; by 4096³ there are 512 and the
un-overlapped load latency has more room to accumulate.

> Note on locked clocks: these medians are ~15% below an earlier unlocked draft
> because the 3090 boosts to ~2 GHz. Pinning the SM clock at 1695 MHz is what makes
> the CUDA-vs-Mojo comparison fair: the harness runs the CUDA variant before the Mojo
> one within each kernel, so on an *unlocked* card the compute-heavy 4096³ Mojo run
> lands on a hotter, down-clocked GPU and looks ~13% slower than it is. Locking
> removes that ordering bias entirely (see `results/run-env.txt`).

## Run

```bash
nvcc -O3 -std=c++20 -arch=sm_86 -lcublas -o /tmp/mm cuda/matmul.cu && /tmp/mm --header
.venv/bin/mojo run mojo/matmul.mojo
```
