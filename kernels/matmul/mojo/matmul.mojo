# Single-precision C = A x B (M x K)*(K x N), row-major, on Ampere (sm_86), Mojo.
#
# Variants:
#   tiled     - 32x32 shared-memory tiles, one output element per thread.
#   regblock  - 128x128 block tile, BK=8, each thread computes an 8x8 register
#               tile (mirrors the CUDA regblock kernel). The optimized one.
#
# Matmul is compute-bound; the metric that matters is gflops (2*M*N*K).
# Correctness is checked against a CPU dot-product on a deterministic subset of
# output cells (a full CPU matmul at 4096^3 would take minutes).
from std.math import ceildiv
from std.sys import has_accelerator
from std.sys.info import has_nvidia_gpu_accelerator
from std.gpu import thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from max.gpu.memory import AddressSpace

comptime dtype = DType.float32

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

# ---- tiled (32x32) --------------------------------------------------------
comptime T = 32

def mm_tiled[LT: TensorLayout](
    A: TileTensor[dtype, LT, MutAnyOrigin],
    B: TileTensor[dtype, LT, MutAnyOrigin],
    C: TileTensor[dtype, LT, MutAnyOrigin],
    M: Int64, N: Int64, K: Int64,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    var As = stack_allocation[dtype, address_space=AddressSpace.SHARED](row_major[T, T]())
    var Bs = stack_allocation[dtype, address_space=AddressSpace.SHARED](row_major[T, T]())
    var ty = thread_idx.y
    var tx = thread_idx.x
    var row = block_idx.y * T + ty
    var col = block_idx.x * T + tx
    var acc: Float32 = 0.0
    var nk = Int(K)
    var k0 = 0
    while k0 < nk:
        As[ty, tx] = A[row, k0 + tx]
        Bs[ty, tx] = B[k0 + ty, col]
        barrier()
        comptime for k in range(T):
            acc += rebind[Float32](As[ty, k]) * rebind[Float32](Bs[k, tx])
        barrier()
        k0 += T
    C[row, col] = rebind[C.ElementType](acc)

# ---- regblock: device-derived tile config (see docs/portability.md) -------
# The register tile (TM x TN) is the perf knob that must track the register
# file: 8x8 = 64 accumulators/thread is calibrated for GA102's 64K regs/SM; on a
# smaller register file that spills to local memory and falls off a cliff. It is
# selected HOST-side from has_nvidia_gpu_accelerator() -- is_nvidia_gpu() reports
# CPU at module scope on this toolchain (verified), so it cannot pick the config.
# Both the host launch dims (grid/block) and the kernel's shared-memory + unroll
# bounds read these SAME comptime constants, so they cannot drift apart.
# Invariant the cooperative float4 loader relies on: BM == BN and, since one
# float4 is loaded per thread, NTHREADS == BM*BK//4 == BK*BN//4.
comptime _NV = has_nvidia_gpu_accelerator()
comptime BM = 128 if _NV else 64
comptime BN = 128 if _NV else 64
comptime BK = 8
comptime TM = 8
comptime TN = 8 if _NV else 4                    # 64 accums on NVIDIA, 32 elsewhere
comptime NTHREADS = (BM * BN) // (TM * TN)       # 256 on NVIDIA, 128 elsewhere

def mm_regblock[LT: TensorLayout](
    A: TileTensor[dtype, LT, MutAnyOrigin],
    B: TileTensor[dtype, LT, MutAnyOrigin],
    C: TileTensor[dtype, LT, MutAnyOrigin],
    M: Int64, N: Int64, K: Int64,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    var As = stack_allocation[dtype, address_space=AddressSpace.SHARED](row_major[BK, BM]())
    var Bs = stack_allocation[dtype, address_space=AddressSpace.SHARED](row_major[BK, BN]())

    var tid = thread_idx.x
    var threadCol = tid % (BN // TN)             # 0..15
    var threadRow = tid // (BN // TN)            # 0..15
    var cRow = block_idx.y
    var cCol = block_idx.x
    var nk = Int(K)
    var ncols = Int(N)

    var results = stack_allocation[dtype](row_major[TM, TN]()).fill(0)
    var regM = stack_allocation[dtype](row_major[TM]())
    var regN = stack_allocation[dtype](row_major[TN]())

    # Base pointers for float4-vectorized global loads (mirrors the CUDA kernel).
    var pA = A.ptr
    var pB = B.ptr

    var k0 = 0
    while k0 < nk:
        # Vectorized cooperative load. BM*BK/4 == BK*BN/4 == NTHREADS (256), so
        # each thread issues exactly one coalesced float4 load per tile.
        # As tile (BM x BK): float4 along K, scattered transposed into As[BK][BM].
        var am = tid // (BK // 4)           # 0..127  row within the A tile
        var akk = (tid % (BK // 4)) * 4     # 0 or 4  K offset
        var av = pA.unsafe_load[width=4]((cRow * BM + am) * nk + k0 + akk)
        As[akk + 0, am] = rebind[As.ElementType](av[0])
        As[akk + 1, am] = rebind[As.ElementType](av[1])
        As[akk + 2, am] = rebind[As.ElementType](av[2])
        As[akk + 3, am] = rebind[As.ElementType](av[3])
        # Bs tile (BK x BN): float4 along N, stored contiguously into Bs[BK][BN].
        var bk = tid // (BN // 4)           # 0..7   K row
        var bn = (tid % (BN // 4)) * 4      # 0,4,..,124  N offset
        var bv = pB.unsafe_load[width=4]((k0 + bk) * ncols + cCol * BN + bn)
        Bs[bk, bn + 0] = rebind[Bs.ElementType](bv[0])
        Bs[bk, bn + 1] = rebind[Bs.ElementType](bv[1])
        Bs[bk, bn + 2] = rebind[Bs.ElementType](bv[2])
        Bs[bk, bn + 3] = rebind[Bs.ElementType](bv[3])
        barrier()

        comptime for k in range(BK):
            comptime for i in range(TM):
                regM[i] = As[k, threadRow * TM + i]
            comptime for j in range(TN):
                regN[j] = Bs[k, threadCol * TN + j]
            comptime for i in range(TM):
                comptime for j in range(TN):
                    var cur = rebind[Float32](results[i, j])
                    var a = rebind[Float32](regM[i])
                    var b = rebind[Float32](regN[j])
                    results[i, j] = rebind[results.ElementType](cur + a * b)
        barrier()
        k0 += BK

    comptime for i in range(TM):
        comptime for j in range(TN):
            var gr = cRow * BM + threadRow * TM + i
            var gc = cCol * BN + threadCol * TN + j
            C[gr, gc] = rebind[C.ElementType](results[i, j])

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var sizes = [1024, 2048, 4096]

    for si in range(len(sizes)):
        var Nsz = sizes[si]
        var M = Nsz
        var N = Nsz
        var K = Nsz
        var layout = row_major(Nsz, Nsz)

        var a_buf = ctx.enqueue_create_buffer[dtype](M * K)
        var b_buf = ctx.enqueue_create_buffer[dtype](K * N)
        var c_buf = ctx.enqueue_create_buffer[dtype](M * N)
        var hA = ctx.enqueue_create_host_buffer[dtype](M * K)
        var hB = ctx.enqueue_create_host_buffer[dtype](K * N)
        ctx.synchronize()

        for i in range(M * K):
            hA[i] = Float32((i % 17) - 8) * 0.05
        for i in range(K * N):
            hB[i] = Float32((i % 23) - 11) * 0.04
        ctx.enqueue_copy(dst_buf=a_buf, src_buf=hA)
        ctx.enqueue_copy(dst_buf=b_buf, src_buf=hB)

        var A = TileTensor(a_buf, layout)
        var B = TileTensor(b_buf, layout)
        var C = TileTensor(c_buf, layout)

        var flops = 2.0 * Float64(M) * Float64(N) * Float64(K)
        var bytes = (Float64(M) * Float64(K) + Float64(K) * Float64(N)
                     + Float64(M) * Float64(N)) * 4.0

        comptime k_tiled = mm_tiled[type_of(layout)]
        comptime k_reg = mm_regblock[type_of(layout)]

        # subset correctness check against CPU dot products (inlined below)
        comptime WARMUP = 5
        comptime REPS = 30

        # tiled
        def run_tiled(cc: DeviceContext) raises {A, B, C, M, N, K}:
            cc.enqueue_function[k_tiled](A, B, C, Int64(M), Int64(N), Int64(K),
                grid_dim=(ceildiv(N, T), ceildiv(M, T)), block_dim=(T, T))
        run_tiled(ctx)
        ctx.synchronize()
        var ok_t: Int
        var num_t: Float64 = 0.0
        var den_t: Float64 = 0.0
        with c_buf.map_to_host() as g:
            for s in range(256):
                var r = (s * 37) % M
                var c = (s * 53) % N
                var acc: Float64 = 0.0
                for kk in range(K):
                    acc += Float64(hA[r * K + kk]) * Float64(hB[kk * N + c])
                var d = Float64(g[r * N + c]) - acc
                num_t += d * d
                den_t += acc * acc
        ok_t = 1 if (num_t / den_t) ** 0.5 <= 1e-4 else 0
        for _ in range(WARMUP):
            _ = ctx.execution_time(run_tiled, 1)
        var samp_t: List[Float64] = []
        for _ in range(REPS):
            samp_t.append(Float64(ctx.execution_time(run_tiled, 1)) / 1.0e6)
        _isort(samp_t)
        var med_t = _pctile(samp_t, 0.5)
        var p25_t = _pctile(samp_t, 0.25)
        var p75_t = _pctile(samp_t, 0.75)
        print("matmul,mojo,tiled,f32,", M, ",", N, ",", K, ",", med_t, ",", p25_t,
              ",", p75_t, ",", REPS, ",", flops / (med_t * 1.0e6), ",",
              bytes / (med_t * 1.0e6), ",", ok_t, sep="")

        # regblock
        def run_reg(cc: DeviceContext) raises {A, B, C, M, N, K}:
            cc.enqueue_function[k_reg](A, B, C, Int64(M), Int64(N), Int64(K),
                grid_dim=(N // BN, M // BM), block_dim=NTHREADS)
        run_reg(ctx)
        ctx.synchronize()
        var ok_r: Int
        var num_r: Float64 = 0.0
        var den_r: Float64 = 0.0
        with c_buf.map_to_host() as g:
            for s in range(256):
                var r = (s * 37) % M
                var c = (s * 53) % N
                var acc: Float64 = 0.0
                for kk in range(K):
                    acc += Float64(hA[r * K + kk]) * Float64(hB[kk * N + c])
                var d = Float64(g[r * N + c]) - acc
                num_r += d * d
                den_r += acc * acc
        ok_r = 1 if (num_r / den_r) ** 0.5 <= 1e-4 else 0
        for _ in range(WARMUP):
            _ = ctx.execution_time(run_reg, 1)
        var samp_r: List[Float64] = []
        for _ in range(REPS):
            samp_r.append(Float64(ctx.execution_time(run_reg, 1)) / 1.0e6)
        _isort(samp_r)
        var med_r = _pctile(samp_r, 0.5)
        var p25_r = _pctile(samp_r, 0.25)
        var p75_r = _pctile(samp_r, 0.75)
        print("matmul,mojo,regblock,f32,", M, ",", N, ",", K, ",", med_r, ",", p25_r,
              ",", p75_r, ",", REPS, ",", flops / (med_r * 1.0e6), ",",
              bytes / (med_r * 1.0e6), ",", ok_r, sep="")
