// Single-precision C = A x B (M x K) * (K x N), row-major, on Ampere (sm_86).
//
// Variants, in increasing order of optimization:
//   naive    - one thread per C element, all reads from global memory.
//   tiled    - 32x32 shared-memory tiles, one output per thread.
//   regblock - 128x128 block tile, BK=8, each thread computes an 8x8 register
//              tile, with float4-vectorized global loads and a transposed As
//              tile for vectorized register reads. The "optimizado a fondo" one.
//   cublas   - cuBLAS SGEMM, the vendor reference / correctness oracle.
//
// Matmul is compute-bound; the metric that matters is gflops (2*M*N*K).
#include "../../../bench/bench.cuh"
#include <vector>
#include <random>
#include <string>
#include <cublas_v2.h>

// ---- naive ---------------------------------------------------------------
__global__ void mm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ---- tiled (32x32 shared memory) -----------------------------------------
template <int T>
__global__ void mm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * T + ty;
    int col = blockIdx.x * T + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += T) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ---- regblock: 128x128 block, 8x8 per thread, vectorized -----------------
// Assumes M,N,K are multiples of the tile sizes (true for the power-of-two
// sweep below). BM=BN=128, BK=8, TM=TN=8, 256 threads/block.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
mm_regblock(const float* A, const float* B, float* C, int M, int N, int K) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int threadsPerBlock = (BM * BN) / (TM * TN);   // 256
    const int threadCol = threadIdx.x % (BN / TN);       // 0..15
    const int threadRow = threadIdx.x / (BN / TN);       // 0..15

    __shared__ float As[BK][BM];   // transposed for vectorized register loads
    __shared__ float Bs[BK][BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Load indices (float4 => stride of 4 columns).
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);
    const int strideB = threadsPerBlock / (BN / 4);

    float results[TM * TN] = {0.0f};
    float regM[TM], regN[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load As tile (BM x BK) transposed into As[BK][BM] via float4.
        float4 a = reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
        As[innerColA * 4 + 0][innerRowA] = a.x;
        As[innerColA * 4 + 1][innerRowA] = a.y;
        As[innerColA * 4 + 2][innerRowA] = a.z;
        As[innerColA * 4 + 3][innerRowA] = a.w;
        // Load Bs tile (BK x BN) straight via float4.
        #pragma unroll
        for (int off = 0; off < BK; off += strideB) {
            reinterpret_cast<float4*>(&Bs[innerRowB + off][innerColB * 4])[0] =
                reinterpret_cast<const float4*>(&B[(innerRowB + off) * N + innerColB * 4])[0];
        }
        __syncthreads();
        A += BK;
        B += BK * N;

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[k][threadRow * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[k][threadCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    results[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(threadRow * TM + i) * N + threadCol * TN + j] = results[i * TN + j];
}

int main(int argc, char** argv) {
    bool header = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--header") header = true;
    if (header) bench_print_header();

    std::vector<int> sizes = {1024, 2048, 4096};

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int Nsz : sizes) {
        int M = Nsz, N = Nsz, K = Nsz;
        long total = (long)M * N;
        std::vector<float> hA((long)M * K), hB((long)K * N), hC(total);
        for (auto& x : hA) x = dist(rng);
        for (auto& x : hB) x = dist(rng);

        float *dA, *dB, *dC, *dRef;
        CUDA_CHECK(cudaMalloc(&dA, (long)M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, (long)K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dRef, total * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), (long)M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), (long)K * N * sizeof(float), cudaMemcpyHostToDevice));

        double flops = 2.0 * M * N * K;
        double bytes = ((double)M * K + (double)K * N + total) * sizeof(float);
        float alpha = 1.0f, beta = 0.0f;

        // Reference via cuBLAS (row-major trick: C = A*B computed as C^T).
        auto cublas_launch = [&](float* out) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                        dB, N, dA, K, &beta, out, N);
        };
        cublas_launch(dRef);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> ref(total);
        CUDA_CHECK(cudaMemcpy(ref.data(), dRef, total * sizeof(float), cudaMemcpyDeviceToHost));

        // Relative L2 (Frobenius) error vs the cuBLAS reference — robust to
        // individual near-zero entries produced by random cancellation.
        auto check = [&]() -> int {
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, total * sizeof(float), cudaMemcpyDeviceToHost));
            double num = 0, den = 0;
            for (long i = 0; i < total; ++i) {
                double d = (double)hC[i] - (double)ref[i];
                num += d * d;
                den += (double)ref[i] * (double)ref[i];
            }
            return std::sqrt(num / den) <= 1e-4;
        };

        // naive (skip at 4096: far too slow to be worth timing every rep)
        if (Nsz <= 2048) {
            dim3 blk(16, 16), grd((N + 15) / 16, (M + 15) / 16);
            auto launch = [&]() { mm_naive<<<grd, blk>>>(dA, dB, dC, M, N, K); };
            double ms = bench_time_ms(launch, 20, 3);
            int correct = check();
            bench_emit("matmul", "naive", "f32", M, N, K, ms,
                       flops / (ms * 1e6), bytes / (ms * 1e6), correct);
        }
        // tiled
        {
            constexpr int T = 32;
            dim3 blk(T, T), grd((N + T - 1) / T, (M + T - 1) / T);
            auto launch = [&]() { mm_tiled<T><<<grd, blk>>>(dA, dB, dC, M, N, K); };
            double ms = bench_time_ms(launch, 30, 5);
            int correct = check();
            bench_emit("matmul", "tiled", "f32", M, N, K, ms,
                       flops / (ms * 1e6), bytes / (ms * 1e6), correct);
        }
        // regblock
        {
            constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
            dim3 blk((BM * BN) / (TM * TN));
            dim3 grd(N / BN, M / BM);
            auto launch = [&]() {
                mm_regblock<BM, BN, BK, TM, TN><<<grd, blk>>>(dA, dB, dC, M, N, K);
            };
            double ms = bench_time_ms(launch, 30, 5);
            int correct = check();
            bench_emit("matmul", "regblock", "f32", M, N, K, ms,
                       flops / (ms * 1e6), bytes / (ms * 1e6), correct);
        }
        // cublas
        {
            auto launch = [&]() { cublas_launch(dC); };
            double ms = bench_time_ms(launch, 30, 5);
            int correct = check();
            bench_emit("matmul", "cublas", "f32", M, N, K, ms,
                       flops / (ms * 1e6), bytes / (ms * 1e6), correct);
        }

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC)); CUDA_CHECK(cudaFree(dRef));
    }
    cublasDestroy(handle);
    return 0;
}
