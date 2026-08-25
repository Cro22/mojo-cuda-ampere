// Sum reduction over a large float32 array on NVIDIA Ampere (sm_86).
//
// Two variants:
//   naive      - classic shared-memory tree reduction, one load per thread,
//                partials finished by a second kernel launch.
//   warp_shfl  - grid-stride vectorized (float4) loads + warp-shuffle block
//                reduction + single atomicAdd of each block's partial. This is
//                the memory-bound "roofline" version.
//
// Reduction is a pure memory-bound kernel: the metric that matters is achieved
// DRAM bandwidth (gbytes_s), not FLOP/s. gflops here counts the N-1 adds.
#include "../../../bench/bench.cuh"
#include <cub/device/device_reduce.cuh>   // vendor reference: cub::DeviceReduce
#include <vector>
#include <random>
#include <string>

static constexpr int BLOCK = 256;

// ---- naive: shared-memory tree -------------------------------------------
__global__ void reduce_naive(const float* __restrict__ in, float* __restrict__ out, long n) {
    __shared__ float s[BLOCK];
    long tid = threadIdx.x;
    long i = (long)blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// ---- warp_shfl: grid-stride + float4 + warp shuffle + atomic --------------
__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__global__ void reduce_warp_shfl(const float* __restrict__ in, float* __restrict__ out, long n) {
    // Grid-stride over float4 to maximize memory-level parallelism.
    float sum = 0.0f;
    long n4 = n / 4;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (long)gridDim.x * blockDim.x) {
        float4 v = in4[i];
        sum += v.x + v.y + v.z + v.w;
    }
    // tail elements not covered by the float4 body
    for (long i = n4 * 4 + (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x)
        sum += in[i];

    // Block reduction: warp-shuffle within each warp, then across warps.
    __shared__ float warp_sums[BLOCK / 32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    sum = warp_reduce_sum(sum);
    if (lane == 0) warp_sums[wid] = sum;
    __syncthreads();
    if (wid == 0) {
        sum = (threadIdx.x < BLOCK / 32) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) atomicAdd(out, sum);
    }
}

int main(int argc, char** argv) {
    bool header = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--header") header = true;
    if (header) bench_print_header();

    // Problem-size sweep (elements). ~4M .. 256M floats.
    std::vector<long> sizes = {1L<<22, 1L<<24, 1L<<26, 1L<<28};

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (long n : sizes) {
        std::vector<float> h(n);
        double ref = 0.0;
        for (long i = 0; i < n; ++i) { h[i] = dist(rng); ref += h[i]; }

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        int blocks_naive = (int)((n + BLOCK - 1) / BLOCK);
        CUDA_CHECK(cudaMalloc(&d_out, blocks_naive * sizeof(float)));

        double bytes = (double)n * sizeof(float);
        double flops = (double)(n - 1);

        // ---- naive ----
        {
            std::vector<float> partials(blocks_naive);
            auto launch = [&]() {
                reduce_naive<<<blocks_naive, BLOCK>>>(d_in, d_out, n);
            };
            BenchStat st = bench_time(launch);
            // finish the partials on the host for correctness (cheap vs kernel)
            CUDA_CHECK(cudaMemcpy(partials.data(), d_out, blocks_naive * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            double got = 0.0; for (float p : partials) got += p;
            int correct = approx_equal(got, ref, 1e-3);
            bench_emit("reduction", "naive", "f32", n, 1, 1, st, flops, bytes, correct);
        }

        // ---- warp_shfl ----
        {
            int blocks = 0, min_grid = 0;
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &blocks,
                       reduce_warp_shfl, 0, 0));
            // Saturate the GPU: a few waves of BLOCK-sized blocks.
            int dev; CUDA_CHECK(cudaGetDevice(&dev));
            int sm_count; CUDA_CHECK(cudaDeviceGetAttribute(&sm_count,
                          cudaDevAttrMultiProcessorCount, dev));
            int grid = sm_count * 32;
            auto launch = [&]() {
                CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));
                reduce_warp_shfl<<<grid, BLOCK>>>(d_in, d_out, n);
            };
            BenchStat st = bench_time(launch);
            float got = 0.0f;
            CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
            int correct = approx_equal(got, ref, 1e-3);
            bench_emit("reduction", "warp_shfl", "f32", n, 1, 1, st, flops, bytes, correct);
        }

        // ---- cub (vendor reference) ----
        // cub::DeviceReduce::Sum is NVIDIA's tuned library reduction; it is the
        // "how close to the vendor?" yardstick for the hand-written kernels.
        {
            float* d_sum;
            CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
            void*  d_temp = nullptr;
            size_t temp_bytes = 0;
            // First call with d_temp == nullptr just sizes the scratch buffer.
            cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_sum, n);
            CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
            auto launch = [&]() {
                cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_sum, n);
            };
            BenchStat st = bench_time(launch);
            float got = 0.0f;
            CUDA_CHECK(cudaMemcpy(&got, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
            int correct = approx_equal(got, ref, 1e-3);
            bench_emit("reduction", "cub", "f32", n, 1, 1, st, flops, bytes, correct);
            CUDA_CHECK(cudaFree(d_temp));
            CUDA_CHECK(cudaFree(d_sum));
        }

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }
    return 0;
}
