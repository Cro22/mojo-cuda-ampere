// Row-wise numerically-stable softmax of an M x N float32 matrix (sm_86).
//
// Two variants:
//   naive   - one block per row, three explicit passes over the row in global
//             memory (max, sum of exp, normalize). Reads the row 3x.
//   online  - one block per row, single streaming pass computing (max, sum)
//             together via the online-softmax combine, block-reduced with warp
//             shuffles, then one normalize pass. Reads the row 2x, half the
//             transcendental traffic of the naive three-pass form.
//
// Softmax is bandwidth-bound: traffic = read + write the matrix = 2*M*N*4 B.
// exp() is the dominant arithmetic; gflops counts ~5 flops/element.
#include "../../../bench/bench.cuh"
#include <vector>
#include <random>
#include <cmath>
#include <string>

static constexpr int BLOCK = 256;

// ---- naive three-pass -----------------------------------------------------
__global__ void softmax_naive(const float* __restrict__ in, float* __restrict__ out,
                              long M, long N) {
    long row = blockIdx.x;
    if (row >= M) return;
    const float* r = in + row * N;
    float* o = out + row * N;
    __shared__ float red[BLOCK];
    int tid = threadIdx.x;

    // pass 1: max
    float m = -INFINITY;
    for (long j = tid; j < N; j += blockDim.x) m = fmaxf(m, r[j]);
    red[tid] = m; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float row_max = red[0]; __syncthreads();

    // pass 2: sum of exp
    float sum = 0.0f;
    for (long j = tid; j < N; j += blockDim.x) sum += __expf(r[j] - row_max);
    red[tid] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float row_sum = red[0]; __syncthreads();

    // pass 3: normalize
    float inv = 1.0f / row_sum;
    for (long j = tid; j < N; j += blockDim.x) o[j] = __expf(r[j] - row_max) * inv;
}

// ---- online single-pass (max,sum) fused -----------------------------------
// Combine two partial (max, sum) reductions.
__inline__ __device__ void combine(float& m, float& s, float m2, float s2) {
    float M = fmaxf(m, m2);
    s = s * __expf(m - M) + s2 * __expf(m2 - M);
    m = M;
}

__inline__ __device__ void warp_reduce_ms(float& m, float& s) {
    for (int off = warpSize / 2; off > 0; off >>= 1) {
        float m2 = __shfl_down_sync(0xffffffffu, m, off);
        float s2 = __shfl_down_sync(0xffffffffu, s, off);
        combine(m, s, m2, s2);
    }
}

__global__ void softmax_online(const float* __restrict__ in, float* __restrict__ out,
                               long M, long N) {
    long row = blockIdx.x;
    if (row >= M) return;
    const float* r = in + row * N;
    float* o = out + row * N;
    int tid = threadIdx.x;

    // Streaming pass: each thread keeps a running (max,sum) over its columns.
    float m = -INFINITY, s = 0.0f;
    for (long j = tid; j < N; j += blockDim.x) {
        float x = r[j];
        float Mn = fmaxf(m, x);
        s = s * __expf(m - Mn) + __expf(x - Mn);
        m = Mn;
    }

    // Block reduction over (m,s) pairs: warp shuffle + shared across warps.
    __shared__ float sm[BLOCK / 32], ss[BLOCK / 32];
    int lane = tid & 31, wid = tid >> 5;
    warp_reduce_ms(m, s);
    if (lane == 0) { sm[wid] = m; ss[wid] = s; }
    __syncthreads();
    if (wid == 0) {
        m = (tid < BLOCK / 32) ? sm[lane] : -INFINITY;
        s = (tid < BLOCK / 32) ? ss[lane] : 0.0f;
        warp_reduce_ms(m, s);
        if (lane == 0) { sm[0] = m; ss[0] = s; }
    }
    __syncthreads();
    float row_max = sm[0], inv = 1.0f / ss[0];

    // normalize
    for (long j = tid; j < N; j += blockDim.x) o[j] = __expf(r[j] - row_max) * inv;
}

int main(int argc, char** argv) {
    bool header = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--header") header = true;
    if (header) bench_print_header();

    // shapes: (rows, cols) holding ~16M elements, sweeping cols.
    std::vector<std::pair<long,long>> shapes = {
        {16384, 1024}, {4096, 4096}, {1024, 16384}};

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

    for (auto [M, N] : shapes) {
        long total = M * N;
        std::vector<float> h(total), ref(total);
        for (long i = 0; i < total; ++i) h[i] = dist(rng);
        // CPU reference softmax
        for (long r = 0; r < M; ++r) {
            const float* row = h.data() + r * N;
            float mx = -INFINITY; for (long j = 0; j < N; ++j) mx = std::max(mx, row[j]);
            double sm = 0; for (long j = 0; j < N; ++j) sm += std::exp(row[j] - mx);
            for (long j = 0; j < N; ++j) ref[r * N + j] = (float)(std::exp(row[j] - mx) / sm);
        }

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h.data(), total * sizeof(float), cudaMemcpyHostToDevice));

        double bytes = 2.0 * total * sizeof(float);
        double flops = 5.0 * total;
        std::vector<float> got(total);

        auto check = [&]() -> int {
            CUDA_CHECK(cudaMemcpy(got.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost));
            double maxrel = 0;
            for (long i = 0; i < total; ++i) {
                double e = std::fabs(got[i] - ref[i]) / (std::fabs(ref[i]) + 1e-6);
                maxrel = std::max(maxrel, e);
            }
            return maxrel <= 1e-3;
        };

        {
            auto launch = [&]() { softmax_naive<<<M, BLOCK>>>(d_in, d_out, M, N); };
            double ms = bench_time_ms(launch);
            int correct = check();
            bench_emit("softmax", "naive", "f32", M, N, 1, ms,
                       flops / (ms * 1e6), bytes / (ms * 1e6), correct);
        }
        {
            auto launch = [&]() { softmax_online<<<M, BLOCK>>>(d_in, d_out, M, N); };
            double ms = bench_time_ms(launch);
            int correct = check();
            bench_emit("softmax", "online", "f32", M, N, 1, ms,
                       flops / (ms * 1e6), bytes / (ms * 1e6), correct);
        }

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }
    return 0;
}
