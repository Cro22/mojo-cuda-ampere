// Shared benchmark utilities for the CUDA side of the shootout.
//
// Single source of truth for the CSV schema emitted by every kernel:
//
//   kernel,impl,variant,dtype,m,n,k,median_ms,p25_ms,p75_ms,n_runs,gflops,gbytes_s,correct
//
// Timing reports a *distribution*, not a single number: each point is measured
// `n_runs` times (after warmup) with CUDA events, and we emit the median plus
// the 25th/75th percentiles (the inter-quartile range). Reporting the IQR is
// what lets the README say "indistinguishable" only when the CUDA and Mojo
// quartile ranges overlap. gflops / gbytes_s are derived from the *median* time
// by each kernel's own work/traffic model.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

// The `impl` column of the CSV. Defaults to "cuda"; the ROCm/HIP build defines
// -DBENCH_IMPL=\"hip\" so the two backends' rows stay distinguishable in one
// results.csv (otherwise every HIP row masquerades as a CUDA row and the plots
// mislabel). Overridable at compile time only — never read at runtime.
#ifndef BENCH_IMPL
#define BENCH_IMPL "cuda"
#endif

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                  \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Per-point timing distribution: median and inter-quartile bounds, in ms.
struct BenchStat {
    double median_ms;
    double p25_ms;
    double p75_ms;
    int    n_runs;
};

// Print the CSV header once (harness passes --header on the first invocation).
static inline void bench_print_header() {
    printf("kernel,impl,variant,dtype,m,n,k,"
           "median_ms,p25_ms,p75_ms,n_runs,gflops,gbytes_s,correct\n");
}

// Linear-interpolated percentile of an ascending-sorted sample vector.
static inline double bench_pct(const std::vector<double>& s, double q) {
    if (s.empty()) return 0.0;
    if (s.size() == 1) return s[0];
    double pos = q * static_cast<double>(s.size() - 1);
    size_t lo = static_cast<size_t>(pos);
    if (lo + 1 >= s.size()) return s.back();
    double frac = pos - static_cast<double>(lo);
    return s[lo] * (1.0 - frac) + s[lo + 1] * frac;
}

// Emit one CSV row. gflops / gbytes_s are derived here from the *median* time so
// every caller uses the exact same denominator.
static inline void bench_emit(const char* kernel, const char* variant,
                              const char* dtype, long m, long n, long k,
                              BenchStat st, double flops, double bytes,
                              int correct) {
    double gflops = flops / (st.median_ms * 1e6);
    double gbytes = bytes / (st.median_ms * 1e6);
    printf("%s,%s,%s,%s,%ld,%ld,%ld,%.6f,%.6f,%.6f,%d,%.3f,%.3f,%d\n",
           kernel, BENCH_IMPL, variant, dtype, m, n, k,
           st.median_ms, st.p25_ms, st.p75_ms, st.n_runs, gflops, gbytes, correct);
    fflush(stdout);
}

// Time a callable (lambda taking no args, launching one kernel iteration) with
// CUDA events. Collects `reps` per-iteration samples after `warmup` untimed
// iterations and returns their median / p25 / p75. Median (not min) so the
// number is a robust central estimate, and the IQR exposes run-to-run spread
// instead of hiding it behind a single lucky minimum.
template <typename F>
static BenchStat bench_time(F&& launch, int reps = 30, int warmup = 5) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> t;
    t.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        t.push_back(static_cast<double>(ms));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(t.begin(), t.end());
    return BenchStat{ bench_pct(t, 0.50), bench_pct(t, 0.25),
                      bench_pct(t, 0.75), reps };
}

// Relative-error correctness check against a reference scalar.
static inline int approx_equal(double got, double ref, double rel_tol) {
    double denom = (ref != 0.0) ? (ref < 0 ? -ref : ref) : 1.0;
    double err = (got - ref);
    if (err < 0) err = -err;
    return (err / denom) <= rel_tol;
}
