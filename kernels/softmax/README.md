# Softmax (row-wise)

Numerically-stable softmax over each row of an `M × N` `float32` matrix:
`y = exp(x - max) / Σ exp(x - max)`. **Memory-bound**, traffic is read + write of
the matrix (`2·4·M·N` bytes), with `exp` as the dominant arithmetic.

## Variants

| variant | passes over the row | idea |
|---------|--------------------|------|
| `naive` | 3 | pass 1 max, pass 2 Σexp, pass 3 normalize, each a full read. |
| `online` | 2 | one streaming pass keeps a running `(max, sum)` via the **online-softmax combine**; a warp-shuffle reduction merges the per-thread pairs; a second pass normalizes. |

The online combine of two partials `(m₁,s₁)`, `(m₂,s₂)`:

```
M = max(m₁, m₂)
s = s₁·exp(m₁-M) + s₂·exp(m₂-M)
```

One block handles one row; threads stride across the columns.

## CUDA ↔ Mojo mapping

Both variants are implemented on both sides, so the naive-vs-online comparison is
symmetric across languages.

| CUDA | Mojo |
|------|------|
| pairwise `__shfl_down_sync` on `(m,s)` (online) | `warp.shuffle_down` on `m` and `s`, same combine |
| shared `red[BLOCK]` log-step tree (naive) | shared `TileTensor` + `stride >>= 1` tree, same shape |
| `__expf` | `std.math.exp` (lowers to the device intrinsic) |
| `__shared__` staging for cross-warp merge | shared `TileTensor` (`AddressSpace.SHARED`) |

## Results (GB/s, clocks locked @1695 MHz, median of 30)

| shape | CUDA online | Mojo online | Mojo/CUDA | CUDA naive | verdict |
|-------|------------:|------------:|:---------:|-----------:|---------|
| 16384×1024 | 730 | 713 | 97.6% | 804 | **indistinguishable** (IQRs overlap) |
| 4096×4096  | 570 | 556 | 97.5% | 488 | CUDA +2% (just outside IQR) |
| 1024×16384 | 535 | 498 | 93.2% | 406 | CUDA +7% (real gap) |

Two things fall out of the locked-clock medians. First, on the square and
row-heavy shapes CUDA and Mojo are a **dead heat**: the online kernels are the same
kernel expressed twice and they land on top of each other. The only real gap is the
very wide 1024×16384 shape, where a shorter kernel makes the per-block `(max,sum)`
reduction a larger fraction of the runtime and CUDA's ~7% edge survives the IQR.

Second, "fewer passes" is not automatically faster. On the **narrow** 16384×1024
rows the naive three-pass kernel (804) actually *beats* online (730): with only 1024
columns the extra read is cheap next to the online combine's per-thread `exp`
bookkeeping. `online` only pays off once the row is **wide** enough (4096, 16384) to
amortize the streaming reduction, where it leads by 15 to 30%. The **Mojo** naive
variant reproduces this crossover (its naive edges its own online on 16384×1024 and
loses on the wider shapes), which confirms the effect is algorithmic, not a
language artifact. The locked-clock Mojo-naive column is a pending re-measure; the
CSV currently carries Mojo `online` only, and the naive numbers behind this note
were taken unlocked, so they are not published as results.

`torch.softmax` fills the vendor column when the optional torch build is installed
(`bench/vendor_softmax.py`); see the repo README setup step.

## Run

```bash
nvcc -O3 -std=c++20 -arch=sm_86 -o /tmp/sm cuda/softmax.cu && /tmp/sm --header
.venv/bin/mojo run mojo/softmax.mojo
```
