# Roofline: RTX 3090 (GA102, sm_86)

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

### A caveat on the 35.6 TFLOP/s fp32 peak

That number is the *marketing* peak and it assumes a pure-FMA instruction stream.
On GA102 (Ampere) the doubled fp32 rate comes from a datapath that each SM
sub-partition **shares between fp32 and INT32**: one of the two pipes can issue
either fp32 *or* int32, but not both in the same cycle. Turing had a dedicated
INT32 pipe; Ampere gave that second pipe an fp32 capability, which is exactly why
the fp32 peak doubled, and why it only materializes when almost nothing else is
competing for that pipe.

Real kernels do issue INT32: address arithmetic, loop counters, index math for
shared-memory and register-tile addressing. Every INT32 instruction that lands on
the shared pipe is an fp32 slot not taken, so the **sustainable** fp32 ceiling for
a kernel with non-trivial index arithmetic is **below** 35.6 TFLOP/s. The
practical consequence for this repo: a matmul reported at "55% of 35.6 TFLOP/s" is
measured against an idealized denominator it can never fully reach, so its
efficiency against the *achievable* fp32 rate is meaningfully higher than 55%.
This is one reason cuBLAS leans on FMA-dense inner loops (and, for other dtypes,
tensor cores) that keep the INT32 traffic per flop as low as possible.

## Where each kernel sits

| kernel | arithmetic intensity (FLOP/byte) | bound | ceiling |
|--------|----------------------------------|-------|---------|
| reduction | `(N-1) / 4N` ≈ **0.25** | memory (far left) | 936 GB/s |
| softmax | `~5MN / 8MN` ≈ **0.6** | memory | 936 GB/s |
| matmul (naive/tiled, per-DRAM) | grows with tile reuse | → compute | 35.6 TFLOP/s |
| matmul (regblock, effective) | ≫ 38 (operands live in registers/shared) | **compute** | 35.6 TFLOP/s |

Reduction and softmax have AI far below the ridge point, they are hard memory-bound
kernels, and the only knob is **achieved bandwidth**. Matmul's *global* AI depends
entirely on tiling: the naive kernel re-reads operands `O(N)` times so its effective
AI is low and it stalls on memory; register blocking raises reuse until the kernel
is limited by fp32 FMA throughput.

## Measured vs ceiling

All numbers below are **medians of 30 samples with clocks locked at SM 1695 MHz**
(so numerator and denominator share the same clock; see the note above on why that
makes the fp32 percentages honest rather than flattering).

Bandwidth-bound kernels (fraction of 936 GB/s):

| kernel / variant | GB/s | % of peak |
|------------------|-----:|----------:|
| reduction · CUDA warp_shfl (256M) | 889 | **95%** |
| reduction · Mojo warp_shfl (256M) | 895 | 96% |
| reduction · CUB `DeviceReduce` (256M) | 895 | 96% |
| softmax · CUDA online (16384×1024) | 722 | 77% |
| softmax · Mojo online (16384×1024) | 728 | 78% |

The three reduction rows are within ~6 GB/s of each other, i.e. the hand-written
kernels (both languages) are indistinguishable from NVIDIA's own library at the bus.

Compute-bound kernel (fraction of 35.6 TFLOP/s, the *idealized* fp32 peak):

| kernel / variant (4096³) | GFLOP/s | % of fp32 peak |
|--------------------------|--------:|---------------:|
| cuBLAS | 23 385 | **66%** |
| CUDA regblock | 17 370 | 49% |
| Mojo regblock | 13 780 | 39% |
| CUDA / Mojo tiled | ~2 500 | ~7% |

(At 1024³ the Mojo regblock reaches 8 497 GFLOP/s vs CUDA's 9 279 (**92%**), where
the smaller working set makes the vectorized loads count for more; the gap widens to
~80% at 2048³/4096³.)

These percentages are lower than an earlier draft claimed (which read "55% / 76%")
because that draft divided a *boosted* ~2 GHz numerator by a *1695 MHz* denominator.
With the clock pinned on both sides the honest figure is 49% (CUDA) / 66% (cuBLAS)
of the idealized peak, and, per the FP32/INT32 caveat above, meaningfully higher
against the peak the kernel can actually sustain.

## Reading the gaps

- **Reduction** saturates the bus in both languages (95-96%), and matches
  `cub::DeviceReduce` to within ~6 GB/s. The `float4`-vectorized grid-stride loop
  issues wide, fully-coalesced transactions; the warp-shuffle reduction adds
  negligible overhead. The Mojo kernel uses the same trick
  (`input.ptr.unsafe_load[width=4]`) and their IQRs overlap, so at 256M the two are
  statistically indistinguishable. A scalar Mojo load (the first cut) sat ~10 points
  lower, which is exactly the cost of not vectorizing.

- **Softmax** cannot reach the reduction's efficiency because it both reads *and*
  writes the matrix and evaluates `exp` twice per element; the online variant wins
  on wide rows by reading the row 2x instead of 3x. On 16384x1024 the two languages
  are indistinguishable (728 vs 722 GB/s, overlapping IQRs); only the very wide
  1024x16384 shape opens a real ~8% gap, where a shorter kernel makes the per-block
  reduction overhead a larger fraction of the runtime. For narrow rows the naive
  three-pass version can even edge the online one ahead.

- **Matmul** is the widest gap. cuBLAS uses hand-tuned SASS, larger tiles, double
  buffering and (for other dtypes) tensor cores; the register-blocked kernel here is
  a clean textbook implementation at ~49% of the idealized peak. Both the CUDA and
  Mojo kernels stage A/B through `float4`-vectorized global loads and a transposed
  `As` tile; the compute inner loop (8x8 register tile) is identical. That parity
  brings Mojo to **92% of CUDA at 1024³** and **~80% at 2048³/4096³**. The residual
  gap is **double buffering** (overlapping the next tile's loads with the current
  tile's FMAs) and instruction scheduling: the next optimization on the Mojo side,
  and where the vendor SASS still wins.

The headline: on **memory-bound** kernels the two languages are statistically
indistinguishable and both near the bus limit; on **compute-bound** matmul, once both
vectorize their loads, Mojo is 80 to 92% of the hand-tuned CUDA kernel, and the rest
is scheduling craft (double buffering), not a language limitation.
