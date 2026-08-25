# Softmax (row-wise)

Numerically-stable softmax over each row of an `M × N` `float32` matrix:
`y = exp(x - max) / Σ exp(x - max)`. **Memory-bound** — traffic is read + write of
the matrix (`2·4·M·N` bytes) — with `exp` as the dominant arithmetic.

## Variants

| variant | passes over the row | idea |
|---------|--------------------|------|
| `naive` | 3 | pass 1 max, pass 2 Σexp, pass 3 normalize — each a full read. |
| `online` | 2 | one streaming pass keeps a running `(max, sum)` via the **online-softmax combine**; a warp-shuffle reduction merges the per-thread pairs; a second pass normalizes. |

The online combine of two partials `(m₁,s₁)`, `(m₂,s₂)`:

```
M = max(m₁, m₂)
s = s₁·exp(m₁-M) + s₂·exp(m₂-M)
```

One block handles one row; threads stride across the columns.

## CUDA ↔ Mojo mapping

| CUDA | Mojo |
|------|------|
| pairwise `__shfl_down_sync` on `(m,s)` | `warp.shuffle_down` on `m` and `s`, same combine |
| `__expf` | `std.math.exp` (lowers to the device intrinsic) |
| `__shared__` staging for cross-warp merge | shared `TileTensor` (`AddressSpace.SHARED`) |

## Results

| shape | CUDA online | Mojo online | CUDA naive |
|-------|------------:|------------:|-----------:|
| 16384×1024 | 771 GB/s | 674 GB/s | 800 GB/s |
| 4096×4096  | 585 | 536 | 504 |
| 1024×16384 | 558 | 504 | 421 |

`online` wins on **wide** rows (fewer reads); on **narrow** rows (1024) the
per-block pair-reduction overhead lets the naive three-pass version edge ahead —
a nice illustration that "fewer passes" only pays once the row is big enough to
amortize the reduction. Mojo tracks CUDA within ~5–15%.

## Run

```bash
nvcc -O3 -std=c++20 -arch=sm_86 -o /tmp/sm cuda/softmax.cu && /tmp/sm --header
.venv/bin/mojo run mojo/softmax.mojo
```
