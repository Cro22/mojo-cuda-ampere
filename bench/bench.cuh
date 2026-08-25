// Shared benchmark utilities for the CUDA side of the shootout.
//
// Single source of truth for the CSV schema emitted by every kernel:
//
//   kernel,impl,variant,dtype,m,n,k,time_ms,gflops,gbytes_s,correct
//
// time_ms is the *minimum* per-iteration time over `reps` timed iterations
// (steady-state, after warmup), measured with CUDA events. gflops / gbytes_s
// are derived from that time by each kernel's own work/traffic model.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                  \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Print the CSV header once (harness passes --header on the first invocation).
static inline void bench_print_header() {
    printf("kernel,impl,variant,dtype,m,n,k,time_ms,gflops,gbytes_s,correct\n");
}

// Emit one CSV row.
static inline void bench_emit(const char* kernel, const char* variant,
                              const char* dtype, long m, long n, long k,
                              double time_ms, double gflops, double gbytes_s,
                              int correct) {
    printf("%s,cuda,%s,%s,%ld,%ld,%ld,%.6f,%.3f,%.3f,%d\n", kernel, variant,
           dtype, m, n, k, time_ms, gflops, gbytes_s, correct);
    fflush(stdout);
}

// Time a callable (lambda taking no args, launching one kernel iteration) with
// CUDA events. Returns the minimum time in milliseconds over `reps` iterations
// after `warmup` untimed iterations.
template <typename F>
static double bench_time_ms(F&& launch, int reps = 50, int warmup = 10) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    double best = DBL_MAX;
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return best;
}

// Relative-error correctness check against a reference scalar.
static inline int approx_equal(double got, double ref, double rel_tol) {
    double denom = (ref != 0.0) ? (ref < 0 ? -ref : ref) : 1.0;
    double err = (got - ref);
    if (err < 0) err = -err;
    return (err / denom) <= rel_tol;
}
