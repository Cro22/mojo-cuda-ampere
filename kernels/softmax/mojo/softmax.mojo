# Row-wise numerically-stable softmax of an M x N float32 matrix (sm_86), Mojo.
#
# Mirrors the CUDA `online` variant: one block per row, a single streaming pass
# computing the running (max, sum) with the online-softmax combine, block-reduced
# over (max,sum) pairs via warp shuffles, then one normalize pass.
#
# Softmax is bandwidth-bound: traffic = read + write matrix = 2*M*N*4 B.
from std.math import exp
from std.sys import has_accelerator
from std.gpu import thread_idx, block_idx, block_dim, lane_id, WARP_SIZE
from std.gpu.primitives import warp
from std.bit import log2_floor
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from max.gpu.memory import AddressSpace

comptime dtype = DType.float32

# ---- device-derived launch config (see docs/portability.md) -----------------
# BLOCK is the only host-side knob; NWARPS and the warp-shuffle tree depth are
# derived from WARP_SIZE *inside* the kernel, where the target is the GPU. On
# NVIDIA (WARP_SIZE=32) this reproduces NWARPS=8 and a 5-step shuffle tree; on a
# 64-lane wavefront it becomes NWARPS=4 and a 6-step tree, instead of silently
# reducing over only the first 32 lanes as the old hardcoded range(5) did.
comptime BLOCK = 256

# --- timing statistics: median + inter-quartile range over N samples ---------
def _isort(mut s: List[Float64]):
    for i in range(1, len(s)):
        var key = s[i]
        var j = i - 1
        while j >= 0 and s[j] > key:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = key

def _pctile(s: List[Float64], q: Float64) -> Float64:
    var nm1 = len(s) - 1
    if nm1 <= 0:
        return s[0]
    var pos = q * Float64(nm1)
    var lo = Int(pos)
    if lo >= nm1:
        return s[nm1]
    var frac = pos - Float64(lo)
    return s[lo] * (1.0 - frac) + s[lo + 1] * frac

# online-softmax combine of two partial (max, sum) reductions
@always_inline
def combine(m: Float32, s: Float32, m2: Float32, s2: Float32) -> Tuple[Float32, Float32]:
    var mx = max(m, m2)
    return (mx, s * exp(m - mx) + s2 * exp(m2 - mx))

@always_inline
def warp_reduce_ms(m0: Float32, s0: Float32) -> Tuple[Float32, Float32]:
    comptime DEPTH = log2_floor(WARP_SIZE)      # 5 on a 32-lane warp, 6 on 64
    var m = m0
    var s = s0
    comptime for i in range(DEPTH):             # offsets WARP_SIZE/2 .. 1
        var off = UInt32(1 << (DEPTH - 1 - i))
        var m2 = warp.shuffle_down(m, off)
        var s2 = warp.shuffle_down(s, off)
        var r = combine(m, s, m2, s2)
        m = r[0]
        s = r[1]
    return (m, s)

# Naive three-pass baseline (max, sum of exp, normalize), block-reduced through a
# shared-memory log-step tree. Mirrors the CUDA softmax_naive: reads the row 3x,
# the baseline the online variant is measured against. One block per row.
def softmax_naive[LT: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin],
    M: Int64,
    N: Int64,
):
    comptime assert input.flat_rank == 2 and output.flat_rank == 2
    var row = block_idx.x
    if row >= Int(M):
        return
    var tid = thread_idx.x
    var ncols = Int(N)
    var red = stack_allocation[dtype, address_space=AddressSpace.SHARED](row_major[BLOCK]())

    # pass 1: row max
    var m: Float32 = -1.0e30
    var j = tid
    while j < ncols:
        m = max(m, rebind[Float32](input[row, j]))
        j += block_dim.x
    red[tid] = rebind[red.ElementType](m)
    barrier()
    var stride = BLOCK // 2
    while stride > 0:
        if tid < stride:
            red[tid] = rebind[red.ElementType](
                max(rebind[Float32](red[tid]), rebind[Float32](red[tid + stride])))
        barrier()
        stride //= 2
    var row_max = rebind[Float32](red[0])
    barrier()

    # pass 2: sum of exp(x - row_max)
    var s: Float32 = 0.0
    j = tid
    while j < ncols:
        s += exp(rebind[Float32](input[row, j]) - row_max)
        j += block_dim.x
    red[tid] = rebind[red.ElementType](s)
    barrier()
    stride = BLOCK // 2
    while stride > 0:
        if tid < stride:
            red[tid] = rebind[red.ElementType](
                rebind[Float32](red[tid]) + rebind[Float32](red[tid + stride]))
        barrier()
        stride //= 2
    var inv = 1.0 / rebind[Float32](red[0])

    # pass 3: normalize
    j = tid
    while j < ncols:
        output[row, j] = rebind[output.ElementType](
            exp(rebind[Float32](input[row, j]) - row_max) * inv)
        j += block_dim.x

def softmax_online[LT: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin],
    M: Int64,
    N: Int64,
):
    comptime assert input.flat_rank == 2 and output.flat_rank == 2
    var row = block_idx.x
    if row >= Int(M):
        return
    var tid = thread_idx.x
    var ncols = Int(N)

    # streaming pass: per-thread running (max, sum)
    var m: Float32 = -1.0e30
    var s: Float32 = 0.0
    var j = tid
    while j < ncols:
        var x = rebind[Float32](input[row, j])
        var mn = max(m, x)
        s = s * exp(m - mn) + exp(x - mn)
        m = mn
        j += block_dim.x

    # block reduction over (m, s) pairs
    comptime NWARPS = BLOCK // WARP_SIZE         # in-kernel: WARP_SIZE is the GPU's
    var sm = stack_allocation[dtype, address_space=AddressSpace.SHARED](row_major[NWARPS]())
    var ss = stack_allocation[dtype, address_space=AddressSpace.SHARED](row_major[NWARPS]())
    var lane = lane_id()
    var wid = tid // WARP_SIZE
    var wr = warp_reduce_ms(m, s)
    if lane == 0:
        sm[wid] = wr[0]
        ss[wid] = wr[1]
    barrier()

    # Final reduction across the NWARPS partials in warp 0. The per-thread
    # streaming seeds (m=-inf, s=0) live above in `m`/`s`; row_max/row_sum are
    # just the broadcast holders, so declare them at the point they are read
    # (sm[0]/ss[0] after the barrier) rather than seeding a value nothing uses.
    if wid == 0:
        var mm = rebind[Float32](sm[lane]) if lane < NWARPS else Float32(-1.0e30)
        var ssum = rebind[Float32](ss[lane]) if lane < NWARPS else Float32(0.0)
        var fr = warp_reduce_ms(mm, ssum)
        if lane == 0:
            sm[0] = fr[0]
            ss[0] = fr[1]
    barrier()
    var row_max = rebind[Float32](sm[0])
    var row_sum = rebind[Float32](ss[0])

    # normalize
    var inv = 1.0 / row_sum
    var jj = tid
    while jj < ncols:
        output[row, jj] = rebind[output.ElementType](exp(rebind[Float32](input[row, jj]) - row_max) * inv)
        jj += block_dim.x

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var rows_list = [16384, 4096, 1024]
    var cols_list = [1024, 4096, 16384]

    for si in range(len(rows_list)):
        var M = rows_list[si]
        var N = cols_list[si]
        var total = M * N
        var layout = row_major(M, N)

        var in_buf = ctx.enqueue_create_buffer[dtype](total)
        var out_buf = ctx.enqueue_create_buffer[dtype](total)
        var host = ctx.enqueue_create_host_buffer[dtype](total)
        var refh = ctx.enqueue_create_host_buffer[dtype](total)
        ctx.synchronize()

        for i in range(total):
            host[i] = Float32((i % 257) - 128) * 0.03

        # CPU reference softmax per row
        for r in range(M):
            var mx: Float32 = -1.0e30
            for c in range(N):
                mx = max(mx, host[r * N + c])
            var sm2: Float64 = 0.0
            for c in range(N):
                sm2 += Float64(exp(host[r * N + c] - mx))
            for c in range(N):
                refh[r * N + c] = Float32(Float64(exp(host[r * N + c] - mx)) / sm2)

        ctx.enqueue_copy(dst_buf=in_buf, src_buf=host)
        var input = TileTensor(in_buf, layout)
        var output = TileTensor(out_buf, layout)

        comptime WARMUP = 5
        comptime REPS = 30

        # ---- naive (three-pass) ----
        comptime kern_n = softmax_naive[type_of(layout)]
        def run_n(c: DeviceContext) raises {input, output, M, N}:
            c.enqueue_function[kern_n](input, output, Int64(M), Int64(N),
                grid_dim=M, block_dim=BLOCK)
        run_n(ctx)
        ctx.synchronize()
        var num_n: Float64 = 0.0
        var den_n: Float64 = 0.0
        with out_buf.map_to_host() as g:
            for i in range(total):
                var d = Float64(g[i]) - Float64(refh[i])
                num_n += d * d
                den_n += Float64(refh[i]) * Float64(refh[i])
        var correct_n = 1 if (num_n / den_n) ** 0.5 <= 1e-4 else 0
        for _ in range(WARMUP):
            _ = ctx.execution_time(run_n, 1)
        var samp_n: List[Float64] = []
        for _ in range(REPS):
            samp_n.append(Float64(ctx.execution_time(run_n, 1)) / 1.0e6)
        _isort(samp_n)
        var med_n = _pctile(samp_n, 0.5)
        var p25_n = _pctile(samp_n, 0.25)
        var p75_n = _pctile(samp_n, 0.75)
        print("softmax,mojo,naive,f32,", M, ",", N, ",1,", med_n, ",", p25_n,
              ",", p75_n, ",", REPS, ",", 5.0 * Float64(total) / (med_n * 1.0e6),
              ",", 2.0 * Float64(total) * 4.0 / (med_n * 1.0e6), ",", correct_n, sep="")

        # ---- online (single-pass) ----
        comptime kern = softmax_online[type_of(layout)]

        def run(c: DeviceContext) raises {input, output, M, N}:
            c.enqueue_function[kern](input, output, Int64(M), Int64(N),
                grid_dim=M, block_dim=BLOCK)

        run(ctx)
        ctx.synchronize()

        # correctness: relative L2 over the whole matrix
        var num: Float64 = 0.0
        var den: Float64 = 0.0
        with out_buf.map_to_host() as g:
            for i in range(total):
                var d = Float64(g[i]) - Float64(refh[i])
                num += d * d
                den += Float64(refh[i]) * Float64(refh[i])
        var correct = 1 if (num / den) ** 0.5 <= 1e-4 else 0

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

        var bytes = 2.0 * Float64(total) * 4.0
        var flops = 5.0 * Float64(total)
        var gflops = flops / (median_ms * 1.0e6)
        var gbytes = bytes / (median_ms * 1.0e6)
        print("softmax,mojo,online,f32,", M, ",", N, ",1,", median_ms, ",", p25_ms,
              ",", p75_ms, ",", REPS, ",", gflops, ",", gbytes, ",", correct, sep="")
