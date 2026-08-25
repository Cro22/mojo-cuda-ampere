# Matmul (SGEMM)

`C = A × B`, `(M×K)·(K×N)`, row-major `float32`. **Compute-bound** once tiled — the
metric is GFLOP/s against the 35.6 TFLOP/s fp32 peak. `2·M·N·K` flops.

## Variants

| variant | tile | per-thread | idea |
|---------|------|-----------|------|
| `naive` | — | 1 output | one thread per `C` element, all reads from global memory. |
| `tiled` | 32×32 | 1 output | shared-memory tiles; each element reused 32×. |
| `regblock` | 128×128, `BK=8` | **8×8 register tile** | 256 threads/block, each computing an 8×8 block of `C` from register-resident operands. The optimized kernel. |
| `cublas` | — | — | vendor SGEMM, the reference / correctness oracle (CUDA only). |

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

## Results (GFLOP/s)

| size | CUDA regblock | Mojo regblock | Mojo/CUDA | cuBLAS |
|------|-------------:|-------------:|:---------:|-------:|
| 1024³ | 9 850 | 9 040 | **92%** | 19 400 |
| 2048³ | 17 200 | ~12 500 | ~73% | 25 700 |
| 4096³ | **19 700** | **16 500** | **84%** | 27 000 |

*(Single-run figures; ±few % run-to-run.)* Register blocking is a ~7× jump over the
simple tiled kernel. With `float4`-vectorized loads on **both** sides, Mojo now
reaches 92% of the CUDA kernel at 1024³ and ~84% at 4096³. The residual gap is
**double buffering** (overlapping the next tile's global loads with the current
tile's FMAs) and instruction scheduling — not memory movement, and not the
arithmetic (the 8×8 register inner loop is identical).

> 2048³ dips (~73%) for both languages relative to 1024³/4096³: 2048/128 = 16
> blocks per side → 256 blocks over 82 SMs ≈ 3.1 waves, the worst wave-quantization
> of the three sizes. It is a scheduling artifact of the tile/GPU ratio, not a
> kernel defect.

## Run

```bash
nvcc -O3 -std=c++20 -arch=sm_86 -lcublas -o /tmp/mm cuda/matmul.cu && /tmp/mm --header
.venv/bin/mojo run mojo/matmul.mojo
```
