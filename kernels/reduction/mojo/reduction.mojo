# Sum reduction over a large float32 array on NVIDIA Ampere (sm_86), in Mojo.
#
# Mirrors the CUDA warp_shfl variant: grid-stride load, warp-shuffle block
# reduction (warp.sum), one partial per block, then a second single-block pass
# over the partials. Emits the shared CSV schema:
#   kernel,impl,variant,dtype,m,n,k,time_ms,gflops,gbytes_s,correct
#
# Reduction is memory-bound: the number that matters is gbytes_s.
from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import global_idx, thread_idx, block_idx, block_dim, grid_dim
from std.gpu import lane_id, WARP_SIZE
from std.gpu.primitives import warp
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceAttribute
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from max.gpu.memory import AddressSpace

comptime dtype = DType.float32

# ---- device-derived launch config (see docs/portability.md) -----------------
# BLOCK and the per-SM wave multiplier are host-side knobs. The warp-level count
# (NWARPS) is derived from WARP_SIZE *inside* the kernel, where the compilation
# target is the GPU: at module scope WARP_SIZE is a host default (32) and
# is_nvidia_gpu() reports CPU on this toolchain, so warp counts must not be
# baked here. On NVIDIA (WARP_SIZE=32, BLOCK=256) NWARPS reduces to 8, exactly
# the old hardcoded value; on a 64-lane wavefront it correctly becomes 4.
comptime BLOCK = 256
comptime WAVES_PER_SM = 32          # grid = SM_count * WAVES_PER_SM

# --- timing statistics: median + inter-quartile range over N samples ---------
# Mirrors the CUDA side (bench.cuh): report a distribution, not one number, so
# the write-up can say "indistinguishable" only when the quartile ranges overlap.
def _isort(mut s: List[Float64]):
    for i in range(1, len(s)):
        var key = s[i]
        var j = i - 1
        while j >= 0 and s[j] > key:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = key

def _pctile(s: List[Float64], q: Float64) -> Float64:
    # s must be sorted ascending, non-empty. Linear interpolation between ranks.
    var nm1 = len(s) - 1
    if nm1 <= 0:
        return s[0]
    var pos = q * Float64(nm1)
    var lo = Int(pos)
    if lo >= nm1:
        return s[nm1]
    var frac = pos - Float64(lo)
    return s[lo] * (1.0 - frac) + s[lo + 1] * frac

# Block-level sum reduction of one per-thread value -> returned on lane 0
# (valid only in warp 0). Uses warp.sum + a shared staging array.
@always_inline
def block_reduce_sum(val: Float32) -> Float32:
    comptime NWARPS = BLOCK // WARP_SIZE     # in-kernel: WARP_SIZE is the GPU's
    var s = stack_allocation[dtype, address_space=AddressSpace.SHARED](row_major[NWARPS]())
    var lane = lane_id()
    var wid = thread_idx.x // WARP_SIZE
    var v = warp.sum(val)
    if lane == 0:
        s[wid] = v
    barrier()
    var out: Float32 = 0.0
    if wid == 0:
        var partial = s[lane] if lane < NWARPS else Float32(0.0)
        out = warp.sum(rebind[Float32](partial))
    return out

# Pass 1: grid-stride reduce input -> one partial per block.
def reduce_pass[LT: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    partials: TileTensor[dtype, LT, MutAnyOrigin],
    n: Int64,
):
    comptime assert input.flat_rank == 1 and partials.flat_rank == 1
    var sum: Float32 = 0.0
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(n):
        sum += rebind[Float32](input[i])
        i += stride
    var block_sum = block_reduce_sum(sum)
    if thread_idx.x == 0:
        partials[block_idx.x] = block_sum

# Vectorized pass-1: grid-stride over float4 (n must be a multiple of 4, which
# every power-of-two size in the sweep is). This is what lets Mojo match the
# CUDA warp_shfl kernel's coalesced 128-byte transactions.
def reduce_pass_vec[LT: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    partials: TileTensor[dtype, LT, MutAnyOrigin],
    n4: Int64,
):
    comptime assert input.flat_rank == 1 and partials.flat_rank == 1
    var sum: Float32 = 0.0
    var p = input.ptr
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(n4):
        sum += p.unsafe_load[width=4](i * 4).reduce_add()
        i += stride
    var block_sum = block_reduce_sum(sum)
    if thread_idx.x == 0:
        partials[block_idx.x] = block_sum

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var sizes = [1 << 22, 1 << 24, 1 << 26, 1 << 28]
    # Query SM count from the device so the grid saturates any GPU, not just the
    # 3090 (GA102 = 82). One block per SM x 32 waves keeps every SM busy.
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var grid = sm_count * WAVES_PER_SM

    for si in range(len(sizes)):
        var n = sizes[si]
        var layout = row_major(n)

        var in_buf = ctx.enqueue_create_buffer[dtype](n)
        var part_buf = ctx.enqueue_create_buffer[dtype](grid)
        var out_buf = ctx.enqueue_create_buffer[dtype](1)

        # deterministic-ish fill on host, and compute reference sum
        var host = ctx.enqueue_create_host_buffer[dtype](n)
        ctx.synchronize()
        var ref_sum: Float64 = 0.0
        for i in range(n):
            var x = 1.0 + Float32(i % 13) * 0.1   # all-positive, well-conditioned
            host[i] = x
            ref_sum += Float64(x)
        ctx.enqueue_copy(dst_buf=in_buf, src_buf=host)

        var input = TileTensor(in_buf, layout)
        var partials = TileTensor(part_buf, row_major(grid))
        var out = TileTensor(out_buf, row_major(1))

        comptime k1 = reduce_pass_vec[type_of(layout)]
        comptime k2 = reduce_pass[type_of(row_major(grid))]

        def run(c: DeviceContext) raises {input, partials, out, n, grid}:
            c.enqueue_function[k1](input, partials, Int64(n // 4),
                grid_dim=grid, block_dim=BLOCK)
            c.enqueue_function[k2](partials, out, Int64(grid),
                grid_dim=1, block_dim=BLOCK)

        # warmup + correctness
        run(ctx)
        ctx.synchronize()
        var got: Float32
        with out_buf.map_to_host() as h:
            got = h[0]
        var correct = 1 if abs(Float64(got) - ref_sum) / (abs(ref_sum) + 1.0) <= 1e-3 else 0

        # 30 timed samples after 5 warmup; each execution_time(run, 1) is one
        # iteration's device time. Report median + p25/p75 (matches bench.cuh).
        comptime WARMUP = 5
        comptime REPS = 30
        for _ in range(WARMUP):
            _ = ctx.execution_time(run, 1)
        var samples: List[Float64] = []
        for _ in range(REPS):
            var ns = ctx.execution_time(run, 1)
            samples.append(Float64(ns) / 1.0e6)
        _isort(samples)
        var median_ms = _pctile(samples, 0.5)
        var p25_ms = _pctile(samples, 0.25)
        var p75_ms = _pctile(samples, 0.75)

        var bytes = Float64(n) * 4.0
        var flops = Float64(n - 1)
        var gflops = flops / (median_ms * 1.0e6)
        var gbytes = bytes / (median_ms * 1.0e6)
        print("reduction,mojo,warp_shfl,f32,", n, ",1,1,", median_ms, ",", p25_ms,
              ",", p75_ms, ",", REPS, ",", gflops, ",", gbytes, ",", correct, sep="")
