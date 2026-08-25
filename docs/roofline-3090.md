# Roofline — RTX 3090 (GA102, sm_86)

## Hardware ceilings

| Quantity | Value | Source |
|----------|-------|--------|
| SMs | 82 | queried at runtime |
| CUDA cores (fp32) | 10 496 | 128 × 82 |
| Boost clock | ~1.695 GHz | spec |
| **fp32 peak** | **35.6 TFLOP/s** | 10496 × 1.695 GHz × 2 (FMA) |
| TF32 tensor-core peak | ~71 TFLOP/s | (not used here; fp32 CUDA cores only) |
| Memory | 24 GB GDDR6X, 384-bit, 19.5 Gbps | spec |
| **DRAM peak** | **936 GB/s** | 384/8 × 19.5 |
| L2 cache | 6 MB | spec |

The **ridge point** (where a kernel switches from memory-bound to compute-bound)
is `35 580 / 936 ≈ 38 FLOP/byte`. A kernel needs to do ~38 flops per byte of DRAM
traffic before compute becomes the limit.

## Where each kernel sits

| kernel | arithmetic intensity (FLOP/byte) | bound | ceiling |
|--------|----------------------------------|-------|---------|
| reduction | `(N-1) / 4N` ≈ **0.25** | memory (far left) | 936 GB/s |
| softmax | `~5MN / 8MN` ≈ **0.6** | memory | 936 GB/s |
| matmul (naive/tiled, per-DRAM) | grows with tile reuse | → compute | 35.6 TFLOP/s |
| matmul (regblock, effective) | ≫ 38 (operands live in registers/shared) | **compute** | 35.6 TFLOP/s |

Reduction and softmax have AI far below the ridge point — they are hard memory-bound
kernels, and the only knob is **achieved bandwidth**. Matmul's *global* AI depends
entirely on tiling: the naive kernel re-reads operands `O(N)` times so its effective
AI is low and it stalls on memory; register blocking raises reuse until the kernel
is limited by fp32 FMA throughput.

## Measured vs ceiling

Bandwidth-bound kernels (fraction of 936 GB/s):

| kernel / variant | GB/s | % of peak |
|------------------|-----:|----------:|
| reduction · CUDA warp_shfl (256M) | 902 | **96%** |
| reduction · Mojo warp_shfl (256M) | 853 | 91% |
| softmax · CUDA online (16384×1024) | 771 | 82% |
| softmax · Mojo online (16384×1024) | 674 | 72% |

Compute-bound kernel (fraction of 35.6 TFLOP/s):

| kernel / variant (4096³) | GFLOP/s | % of fp32 peak |
|--------------------------|--------:|---------------:|
| cuBLAS | 27 000 | **76%** |
| CUDA regblock | 19 700 | 55% |
| Mojo regblock | 16 500 | 46% |
| CUDA / Mojo tiled | ~2 900 | ~8% |

(At 1024³ the Mojo regblock reaches 9 040 GFLOP/s vs CUDA's 9 850 — **92%** — where
the smaller working set makes the vectorized loads count for more.)

## Reading the gaps

- **Reduction** basically saturates the bus in CUDA (96%). The `float4`-vectorized
  grid-stride loop issues wide, fully-coalesced transactions; the warp-shuffle
  reduction adds negligible overhead. The Mojo kernel uses the same trick
  (`input.ptr.unsafe_load[width=4]`) and lands at 91% of the bus — within ~5% of
  CUDA. A scalar Mojo load (the first cut) sat ~10 points lower, which is exactly
  the cost of not vectorizing.

- **Softmax** cannot reach the reduction's efficiency because it both reads *and*
  writes the matrix and evaluates `exp` twice per element; the online variant wins
  on wide rows by reading the row 2× instead of 3×. For narrow rows the per-block
  reduction overhead dominates and the naive three-pass version can edge ahead.

- **Matmul** is the widest gap. cuBLAS uses hand-tuned SASS, larger tiles, double
  buffering and (for other dtypes) tensor cores; our register-blocked kernel is a
  clean textbook implementation at ~55% of peak. Both the CUDA and Mojo kernels now
  stage A/B through `float4`-vectorized global loads and a transposed `As` tile;
  the compute inner loop (8×8 register tile) is identical. That parity brings Mojo
  to **92% of CUDA at 1024³** and **~84% at 4096³**. The residual gap at large
  sizes is **double buffering** (overlapping the next tile's loads with the current
  tile's FMAs) and instruction scheduling — the next optimization on the Mojo side,
  and where the vendor SASS still wins.

The headline: on **memory-bound** kernels the two languages are within ~10% of each
other and both near the bus limit; on **compute-bound** matmul, once both vectorize
their loads, Mojo is 84–92% of the hand-tuned CUDA kernel, and the rest is
scheduling craft (double buffering), not a language limitation.
