# Reduction (sum)

Sum all elements of a large `float32` array to a single scalar. Pure
**memory-bound** kernel: the metric that matters is achieved DRAM bandwidth, and
the ceiling is 936 GB/s.

## Variants

| variant | idea |
|---------|------|
| `naive` | one load per thread into shared memory, classic log-step tree reduction, one partial per block finished on the host. Baseline. |
| `warp_shfl` | grid-stride load (CUDA: `float4`-vectorized) accumulating per-thread, then a **warp-shuffle** block reduction, then either an atomic (CUDA) or a second single-block pass over the partials (Mojo). |

## CUDA ↔ Mojo mapping

| CUDA | Mojo |
|------|------|
| `__shfl_down_sync(mask, v, off)` | `warp.shuffle_down(v, off)` / `warp.sum(v)` |
| `__shared__ float[]` | `stack_allocation[...address_space=AddressSpace.SHARED]` |
| `atomicAdd(out, v)` | second reduction kernel over partials (avoids a pointer-atomic) |
| `float4` vectorized load (`reinterpret_cast<float4*>`) | `input.ptr.unsafe_load[width=4]` + `.reduce_add()` |
| `cudaDevAttrMultiProcessorCount` | `DeviceAttribute.MULTIPROCESSOR_COUNT` |

Both size the grid to `SM_count × 32` blocks so every SM stays busy across several
waves, independent of array size.

## Results (256M elements, clocks locked @1695 MHz, median of 30)

| impl · variant | GB/s | % of 936 |
|----------------|-----:|---------:|
| CUDA warp_shfl        | 891 | 95% |
| Mojo warp_shfl        | 889 | 95% |
| CUB `DeviceReduce`    | 888 | 95% |
| CUDA naive            | ~258 | ~28% |

The three optimized rows are within ~3 GB/s: the hand-written kernels (both
languages) are **indistinguishable from each other and from NVIDIA's own CUB**
reduction at 256M, all at ~95% of the bus. Their IQRs overlap, so the small ordering
between them is noise, not a result.

## Optimization story

The repo ships the two endpoints as separate variants (`naive` and `warp_shfl`), so
the measured jump is **~258 -> 891 GB/s at 256M, a ~3.5x gain moving the exact same
bytes**. The `warp_shfl` variant folds three independent changes into that gain; the
attribution below is by construction (each addresses a specific stall), not four
separately benchmarked kernels:

1. **Grid-stride, `SM×32` blocks.** The naive kernel launches one block per 256
   elements: a huge grid whose cost is launch and occupancy churn, plus a
   `__syncthreads` barrier storm in the shared-memory tree. Grid-striding lets each
   thread accumulate many elements from a small, SM-sized grid, so every SM stays
   busy across waves independent of `N` and the loads stay coalesced.
2. **Warp-shuffle block reduction** (`__shfl_down` / `warp.sum`) replaces the
   shared-memory tree, removing the barriers and shared round-trips. Once the loads
   dominate, the reduction is essentially free.
3. **`float4` vectorized load** issues 128-bit, fully-coalesced 128-byte
   transactions instead of scalar `float`. This is the step that actually reaches the
   bus: a scalar Mojo load (the first cut) sat ~10 GB/s lower, which is precisely the
   cost of not vectorizing.

The takeaway for a memory-bound kernel: the win is entirely in *how* you issue the
loads and how little you spend between them, not in the arithmetic. To attribute each
step quantitatively you would profile with `ncu` (`bench/profile.sh`) and read
`DRAM Throughput` per variant; that needs GPU counter permissions this WSL setup does
not grant by default.

## Run

```bash
nvcc -O3 -std=c++20 -arch=sm_86 -o /tmp/red cuda/reduction.cu && /tmp/red --header
.venv/bin/mojo run mojo/reduction.mojo
```
