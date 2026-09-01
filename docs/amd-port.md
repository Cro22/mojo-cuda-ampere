# Porting the shootout to AMD (ROCm / HIP) — full-comparison plan

This is the checklist to reproduce the whole CUDA-vs-Mojo experiment on an AMD GPU
in the **AMD Developer Cloud**, as a symmetric **HIP-vs-Mojo** shootout. It records
exactly what has to change, where the vendor APIs differ for real (not just a
rename), and what can only be decided on the box.

Nothing here can be compiled or timed on the NVIDIA dev machine — ROCm is not
present. The point of writing it now is that on the AMD box the work is *run and
debug*, not *design from scratch*.

## Status: what is already pre-wired (done on the NVIDIA box)

The harness auto-detects the platform, so `./bench/run.sh` does the right thing on
either GPU with no hand-editing. Already committed and verified not to regress the
CUDA path (rebuilt + ran on the 3090):

- **`bench/Makefile`** branches on `rocminfo`: empty → the unchanged nvcc/`sm_XX`
  path; a `gfxNNN` → hipcc, `--offload-arch=$(GFX)`, `-lrocblas`, and
  `-DBENCH_IMPL="hip"`. Plus a `make hipify` target that generates the HIP
  sources from the pristine `.cu` (hipify-perl + a sed to repoint the local
  `bench.cuh` include). **Untested until run on ROCm.**
- **`bench/run.sh`** detects the platform once and branches all three GPU blocks
  (clock lock, clock sampler, env record) between `nvidia-smi` and `rocm-smi`.
- **`bench/bench.cuh`** now takes the CSV `impl` column from `BENCH_IMPL`
  (default `"cuda"`, HIP build sets `"hip"`), so both backends' rows coexist in
  one `results.csv`.

So the AMD box flow is: `uv sync` → `make -C bench hipify` (once, then review the
warp/vendor diffs per §3–§4) → `./bench/run.sh`. The remaining work is the parts
that genuinely need the hardware: getting the HIPified C++ to compile clean, the
rocBLAS call, and confirming MAX sees the gfx device.

## 0. What actually changes, and what does not

| Layer | NVIDIA today | AMD target | Nature of the change |
|-------|--------------|------------|----------------------|
| Mojo kernels (`*.mojo`) | portable already | **unchanged** | `WARP_SIZE`/tile derived from device; just needs a MAX+ROCm env |
| `bench/bench.cuh` | `cuda_runtime.h`, `cudaEvent*` | `hip/hip_runtime.h`, `hipEvent*` | mechanical hipify (1:1) **+ one real edit**, see §2 |
| `kernels/*/cuda/*.cu` | nvcc | `hipcc` | hipify; warp intrinsics + atomics need review, see §3 |
| Vendor: matmul | cuBLAS | **rocBLAS** | real API differences, see §4 |
| Vendor: reduction | CUB | **hipCUB / rocPRIM** | include swap, mostly source-compatible |
| Vendor: softmax | torch CUDA | **torch ROCm** | reinstall, no code change |
| `bench/Makefile` | `-arch=sm_XX` via `nvidia-smi` | `--offload-arch=gfxNNN` via `rocminfo` | rewrite arch detection, see §5 |
| `bench/run.sh` | `nvidia-smi` lock/sample/env | `rocm-smi` / `rocminfo` | rewrite the three GPU-query blocks, see §6 |

The Mojo side being untouched is the whole reason this repo parameterized the
kernels (see [portability.md](portability.md)); AMD is the first place that claim
gets *tested* rather than asserted.

## 1. Detect the GPU first (arch is TBD)

On connect, before anything else:

```bash
rocminfo | grep -m1 -E 'gfx[0-9a-f]+'      # -> gfx942 (MI300X), gfx90a (MI210/250), ...
rocm-smi --showproductname
```

Everything below that says `gfxNNN` uses this value. MI300X = `gfx942`,
MI210/MI250 = `gfx90a`. **Wavefront is 64 lanes** on CDNA — this is the number the
Mojo `WARP_SIZE` derivations and the HIP `warpSize` intrinsics must pick up.

## 2. Mojo env on the AMD box (Level-1 sanity check — do this first)

Before porting any C++, prove the Mojo side runs, because it needs zero code
changes and it is the actually-novel result:

```bash
# ROCm is preinstalled on the Dev Cloud images. Recreate the venv there:
uv sync                      # pulls max==26.5.0 + mojo (see pyproject.toml)
.venv/bin/mojo run kernels/reduction/mojo/reduction.mojo   # expect CSV rows, correct=1
.venv/bin/mojo run kernels/softmax/mojo/softmax.mojo
.venv/bin/mojo run kernels/matmul/mojo/matmul.mojo
```

Watch for:
- **Does MAX see the GPU?** `has_accelerator()` must be true. If MAX cannot find
  the ROCm device the kernels abort at the `comptime assert has_accelerator()`.
- **Correctness on 64-lane wavefronts.** The softmax shuffle tree becomes 6 steps
  (`DEPTH = log2_floor(64)`) and `NWARPS = BLOCK//64 = 4`. If a row sum comes back
  wrong, that derivation is the first suspect.
- **matmul takes the non-NVIDIA tile** (64×64, 8×4, 32 accumulators) via
  `has_nvidia_gpu_accelerator() == false`. Expect *correct but untuned* throughput
  — this is the known gap flagged in portability.md, not a bug.

## 3. HIPify the shared header and the three kernels

```bash
# hipify-perl ships with ROCm. Convert in place to .hip.cpp (keep .cu pristine):
for f in bench/bench.cuh kernels/*/cuda/*.cu; do hipify-perl "$f" > "${f%.*}.hip"; done
```

Then review — hipify is ~95% mechanical, these are the 5% that bite:

- **`bench.cuh` → the one non-mechanical edit:** `bench_emit()` hardcodes the CSV
  `impl` column as `"cuda"`. For the AMD runs this must print `"hip"`, or every
  HIP row collides with the CUDA rows in `results.csv` and the plots mislabel.
  Parameterize it or fork a `bench.hip.h`.
- **Warp intrinsics:** `__shfl_down_sync(0xffffffff, v, off)` → HIP's
  `__shfl_down(v, off)` (no mask arg on ROCm). `warpSize` is 64 here, so
  `warpSize/2` starts the shuffle at offset 32 — correct, but confirm the loop
  bound wasn't hardcoded to 16 anywhere on the CUDA side.
- **Atomics:** `atomicAdd(float*)` exists on CDNA but the reduction's
  single-atomic-per-block pattern has different contention characteristics; it
  will *work*, just don't be surprised if the naive/atomic variant reorders vs
  NVIDIA in the results.
- **Block size:** `BLOCK=256` is fine on CDNA (256 threads = 4 wavefronts).

## 4. Vendor references

- **rocBLAS (matmul):** replace `cublasSgemm` + `cublasCreate`/handle with
  `rocblas_sgemm` + `rocblas_create_handle`. Argument order matches BLAS but the
  enum names differ (`rocblas_operation_none`). Link `-lrocblas`. This is the one
  vendor swap with real API surface — budget an hour.
- **hipCUB (reduction):** `<cub/device/device_reduce.cuh>` →
  `<hipcub/hipcub.hpp>`, `cub::DeviceReduce` → `hipcub::DeviceReduce`. Source-
  compatible otherwise. Link/include from the ROCm `hipcub`/`rocprim` packages.
- **torch ROCm (softmax):** `bench/vendor_softmax.py` needs no change; install the
  ROCm torch wheel (`--index-url .../rocm6.x`) into the venv. It is optional and
  the harness already `|| true`s past it if torch is absent.

## 5. `bench/Makefile` — arch detection

Replace the `nvidia-smi`/`nvcc` block with a compiler + arch pair chosen by which
toolchain is present:

```make
# Detect ROCm vs CUDA. On AMD: hipcc + --offload-arch from rocminfo.
GFX := $(shell rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')
ifdef GFX
  CXX     := hipcc
  ARCHFLAG:= --offload-arch=$(GFX)
  VENDOR_MATMUL := -lrocblas
else
  # ... existing nvcc / sm_XX path unchanged ...
endif
```

Keep the NVIDIA path intact so the repo still builds on the 3090; branch on `GFX`.

## 6. `bench/run.sh` — the three GPU-query blocks

`run.sh` touches the GPU in exactly three places, all `nvidia-smi`:

1. **Clock lock** (`nvidia-smi -lgc/-lmc`) → `rocm-smi --setperflevel high` (or
   `--setsclk <level>`). On the Dev Cloud you may not have privilege to pin
   clocks; if so, set `NOLOCK=1`-equivalent and let the IQR absorb DVFS, exactly
   as the WSL path already does. **This is the honest fallback, not a failure.**
2. **Clock sampler** (`nvidia-smi --query-gpu=clocks.sm,clocks.mem -lms 250`) →
   `rocm-smi --showgpuclocks` polled in a loop (rocm-smi has no built-in `-lms`).
3. **Env record** (name/driver/nvcc) → `rocm-smi --showproductname`,
   `rocminfo` for the gfx arch, `hipcc --version`.

Gate all three on the same `GFX`/`ROCM` detection so one `run.sh` serves both
platforms.

## 7. What "done" looks like

A `results/results.csv` on the AMD box with `impl ∈ {hip, mojo}` rows for all
three kernels, correctness=1 everywhere, plus a `run-env.txt` naming the gfx arch
and whether clocks were pinned. Then the README gets a second results table (AMD),
and portability.md finally gets to replace *"unmeasured"* with real MI-series
throughput for the non-NVIDIA Mojo tile.

## Open questions that only the box can answer

- Can we pin clocks on the Dev Cloud, or is it DVFS-only? (Changes IQR width.)
- Does MAX 26.5 detect the specific gfx arch, or do we need a newer MAX nightly?
- Is the conservative non-NVIDIA matmul tile catastrophic (register spill) on
  CDNA, or merely untuned? That decides whether the *next* step is autotuning or
  a bug hunt.
