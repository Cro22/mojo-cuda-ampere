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

## Results (256M elements)

| impl · variant | GB/s | % of 936 |
|----------------|-----:|---------:|
| CUDA warp_shfl | 902 | 96% |
| Mojo warp_shfl | 853 | 91% |
| CUDA naive | ~348 | 37% |

Both saturate the bus once the load is `float4`-vectorized — Mojo lands within ~5%
of CUDA. The naive tree reduction shows why the grid-stride + warp-shuffle structure
matters: it moves the same bytes but at roughly a third of the bandwidth.

## Run

```bash
nvcc -O3 -std=c++20 -arch=sm_86 -o /tmp/red cuda/reduction.cu && /tmp/red --header
.venv/bin/mojo run mojo/reduction.mojo
```
