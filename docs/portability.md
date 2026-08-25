# Portability: parameterizing the kernels by device

These kernels were first written and tuned against a single GPU (RTX 3090, GA102,
`sm_86`). That is exactly the situation where architecture assumptions get baked
in without anyone noticing, because there is only ever one architecture to notice
on. This note records where those assumptions were, how each is now derived from
the device instead of hardcoded, and, honestly, where real *performance*
portability is and is not there yet.

The CUDA side is NVIDIA-only by construction, so "one source, many devices" is a
claim that lives entirely on the Mojo side. The one exception on the CUDA side is
the build target (see the last section): a binary compiled for the wrong arch
does not run slowly, it does not launch at all.

## The toolchain fact that shapes the whole design

Mojo has two families of device checks, and they do **not** behave the same at
every scope. Verified by probing this build (Mojo 1.0 / MAX 26.5) on the 3090:

| where | `is_nvidia_gpu()` / `WARP_SIZE` (target checks) | `has_nvidia_gpu_accelerator()` (host check) |
|-------|--------------------------------------------------|---------------------------------------------|
| module / host scope | `is_nvidia_gpu()` -> **false** (target is CPU there); `WARP_SIZE` -> 32 as a **host default**, not a device read | reports the **real** GPU (true on the 3090), and is usable as a `comptime` constant |
| inside a kernel | correct for the GPU: `is_nvidia_gpu()` true, `WARP_SIZE`=32, `log2_floor(WARP_SIZE)`=5 | — |

The consequence is a hard rule:

- **Warp-level quantities** (how many warps in a block, how deep the shuffle tree
  is) are derived from `WARP_SIZE` **inside the kernel**, where the compilation
  target genuinely is the GPU.
- **Anything the host needs** to size a launch or to select a compile-time config
  is taken from `has_*_gpu_accelerator()`, because `is_*_gpu()` at host scope
  silently reports CPU and would send every non-NVIDIA build down the else branch.

## 1. Warp size (was: implicit 32)

Two places assumed a 32-lane warp:

- `NWARPS = BLOCK // 32` (reduction, softmax) sized the shared staging array and
  the cross-warp reduction. It is now `BLOCK // WARP_SIZE`, computed in-kernel. On
  NVIDIA this is still 8; on a 64-lane AMD wavefront it correctly becomes 4 instead
  of over-allocating and reading dead slots.
- The softmax `(max, sum)` warp reduction was a `comptime for i in range(5)` with
  offsets `16,8,4,2,1`, i.e. hardcoded `log2(32)`. On a 64-lane wavefront that
  reduces over only the first 32 lanes and returns a **silently wrong** row sum.
  It is now `comptime for i in range(log2_floor(WARP_SIZE))` with offset
  `1 << (DEPTH-1-i)`: 5 steps on a 32-lane warp, 6 on a 64-lane one.

The reduction kernel's own block reduction already used the warp-size-agnostic
`warp.sum`, and the Mojo reduction is two-pass (partials -> single block), so
unlike the CUDA side it never used an atomic-per-block. Those were already
portable.

## 2. Register / block tile (was: 128x128, 8x8 fixed to GA102)

The regblock matmul's real architecture dependence is register pressure: an 8x8
register tile is 64 accumulators per thread, calibrated for GA102's 64K
registers/SM. On a smaller register file that spills to local memory and falls
off a cliff. The tile is now a **device-derived config selected host-side**:

```mojo
comptime _NV = has_nvidia_gpu_accelerator()
comptime BM = 128 if _NV else 64
comptime BN = 128 if _NV else 64
comptime BK = 8
comptime TM = 8
comptime TN = 8 if _NV else 4          # 64 accumulators on NVIDIA, 32 elsewhere
comptime NTHREADS = (BM * BN) // (TM * TN)
```

Both the host launch dims (`grid_dim`, `block_dim`) and the kernel's shared-memory
layouts and unroll bounds read these *same* comptime constants, so they cannot
drift apart. The cooperative `float4` loader requires `BM == BN` and
`NTHREADS == BM*BK//4`; both configs above satisfy it, so no load code changes
between devices.

**Validation.** On the 3090 the NVIDIA branch reproduces the original numbers
(regblock 1024^3 ~8.5 TFLOP/s, correctness passes). Forcing the *non*-NVIDIA
branch (`_NV = False`, 64x64 tile, 8x4 register tile, 128 threads) on the same
3090 still passes correctness on every size, which shows the parameterization is
genuinely generic and not a cosmetic rename.

## Honest status: is performance portability there?

- **Warp-level correctness portability: yes, and tested.** The warp derivations are
  exercised on NVIDIA and are correct by construction on a 64-lane wavefront.
- **Tile performance portability: the mechanism is there, the tuned numbers are
  not.** The non-NVIDIA tile above is a conservative, register-light *heuristic*.
  It has only been run on NVIDIA hardware (where it is correct); its actual
  occupancy and throughput on AMD/Apple are unmeasured. Picking the *right* tile
  per device is an autotuning problem: sweep candidate configs on the target and
  keep the fastest. A single `mojo run` source cannot auto-select comptime tiles
  purely at compile time here, because host-scope target dispatch is unavailable
  (see the table above); doing it properly means either building per target, or a
  runtime-dispatch-over-precompiled-variants pass that queries the device and
  launches the matching pre-instantiated kernel. That is the same shape as what
  MAX's own kernel library does, and it is the natural next step.

So the parameterization pulls the architecture assumptions out of the kernel and
makes them one named, device-selected config; it does not yet claim the *values*
are optimal anywhere but the card they were tuned on. That is the honest state of
perf portability for these kernels.

## 3. CUDA build target (Makefile)

`bench/Makefile` no longer hardcodes `-arch=sm_86`. It auto-detects the compute
capability of the GPU present and builds for it:

```make
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ARCH    ?= sm_$(if $(DETECTED_CC),$(DETECTED_CC),86)
```

RTX 3090 -> `sm_86`, T4 -> `sm_75`. `compute_cap` is a query, so it works under
WSL. Override explicitly with `make ARCH=sm_80`; it falls back to `sm_86` if
`nvidia-smi` is unavailable (a GPU-less build box).
