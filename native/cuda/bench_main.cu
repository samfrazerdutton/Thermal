// THERMAL kernel lab benchmark harness.
//
// Runs a named kernel for `warmup` (untimed) + `iters` (timed, via CUDA
// events) iterations and emits per-iteration timings as JSON to stdout --
// consumed by the same analysis.baseline statistics engine the Python
// workloads use, so a native kernel's baseline is computed the same way as
// a Python workload's. Correctness is checked against a CPU reference
// before any timing is reported; a kernel that produces wrong output prints
// FAIL and exits non-zero rather than reporting fabricated-looking numbers
// for broken code.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kernels.h"

namespace {

void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

struct Args {
    std::string kernel = "vector_add_naive";
    int n = 1 << 20;
    int iters = 20;
    int warmup = 5;
    int block_size = 256;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (arg == "--kernel") args.kernel = next();
        else if (arg == "--n") args.n = std::atoi(next().c_str());
        else if (arg == "--iters") args.iters = std::atoi(next().c_str());
        else if (arg == "--warmup") args.warmup = std::atoi(next().c_str());
        else if (arg == "--block-size") args.block_size = std::atoi(next().c_str());
    }
    return args;
}

// --- correctness checks (small, fixed sizes -- independent of benchmark n) ---

bool verifyVectorAdd(thermal_kernels::LaunchResult (*fn)(const float*, const float*, float*, int, int)) {
    const int n = 1 << 14;
    std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n, 0.0f);
    float *d_a, *d_b, *d_c;
    checkCuda(cudaMalloc(&d_a, n * sizeof(float)), "malloc a");
    checkCuda(cudaMalloc(&d_b, n * sizeof(float)), "malloc b");
    checkCuda(cudaMalloc(&d_c, n * sizeof(float)), "malloc c");
    cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    fn(d_a, d_b, d_c, n, 256);
    cudaMemcpy(c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    for (int i = 0; i < n; ++i) {
        if (std::fabs(c[i] - 3.0f) > 1e-5f) return false;
    }
    return true;
}

bool verifyReduction(thermal_kernels::LaunchResult (*fn)(const float*, float*, int, int)) {
    const int n = 1 << 16;
    std::vector<float> input(n, 1.0f);
    float expected = static_cast<float>(n);
    float *d_input, *d_result;
    checkCuda(cudaMalloc(&d_input, n * sizeof(float)), "malloc input");
    checkCuda(cudaMalloc(&d_result, sizeof(float)), "malloc result");
    cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    float zero = 0.0f;
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
    fn(d_input, d_result, n, 256);
    float result = 0.0f;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input); cudaFree(d_result);
    return std::fabs(result - expected) < 1e-2f * expected;
}

bool verifyMatmul(thermal_kernels::LaunchResult (*fn)(const float*, const float*, float*, int)) {
    const int n = 128;
    std::vector<float> a(n * n), b(n * n), c(n * n, 0.0f), ref(n * n, 0.0f);
    for (int i = 0; i < n * n; ++i) { a[i] = (i % 7) * 0.1f; b[i] = (i % 5) * 0.2f; }
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) sum += a[row * n + k] * b[k * n + col];
            ref[row * n + col] = sum;
        }
    float *d_a, *d_b, *d_c;
    checkCuda(cudaMalloc(&d_a, n * n * sizeof(float)), "malloc a");
    checkCuda(cudaMalloc(&d_b, n * n * sizeof(float)), "malloc b");
    checkCuda(cudaMalloc(&d_c, n * n * sizeof(float)), "malloc c");
    cudaMemcpy(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    fn(d_a, d_b, d_c, n);
    cudaMemcpy(c.data(), d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    for (int i = 0; i < n * n; ++i) {
        if (std::fabs(c[i] - ref[i]) > 1e-2f * std::max(1.0f, std::fabs(ref[i]))) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    bool correct;
    std::vector<float> times_ms;
    double bytes_moved = 0.0;
    double flops = 0.0;

    if (args.kernel == "vector_add_naive" || args.kernel == "vector_add_grid_stride") {
        auto fn = (args.kernel == "vector_add_naive") ? thermal_kernels::vector_add_naive
                                                        : thermal_kernels::vector_add_grid_stride;
        correct = verifyVectorAdd(fn);
        if (!correct) { printf("{\"kernel\":\"%s\",\"correct\":false}\n", args.kernel.c_str()); return 1; }

        int n = args.n;
        std::vector<float> a(n, 1.0f), b(n, 2.0f);
        float *d_a, *d_b, *d_c;
        checkCuda(cudaMalloc(&d_a, n * sizeof(float)), "malloc a");
        checkCuda(cudaMalloc(&d_b, n * sizeof(float)), "malloc b");
        checkCuda(cudaMalloc(&d_c, n * sizeof(float)), "malloc c");
        cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        for (int i = 0; i < args.warmup; ++i) fn(d_a, d_b, d_c, n, args.block_size);
        for (int i = 0; i < args.iters; ++i) times_ms.push_back(fn(d_a, d_b, d_c, n, args.block_size).kernel_time_ms);

        bytes_moved = 3.0 * n * sizeof(float);  // read a, read b, write c
        flops = static_cast<double>(n);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    } else if (args.kernel == "reduce_naive_atomic" || args.kernel == "reduce_shared_memory") {
        auto fn = (args.kernel == "reduce_naive_atomic") ? thermal_kernels::reduce_naive_atomic
                                                           : thermal_kernels::reduce_shared_memory;
        correct = verifyReduction(fn);
        if (!correct) { printf("{\"kernel\":\"%s\",\"correct\":false}\n", args.kernel.c_str()); return 1; }

        int n = args.n;
        std::vector<float> input(n, 1.0f);
        float *d_input, *d_result;
        checkCuda(cudaMalloc(&d_input, n * sizeof(float)), "malloc input");
        checkCuda(cudaMalloc(&d_result, sizeof(float)), "malloc result");
        cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

        for (int i = 0; i < args.warmup; ++i) {
            float zero = 0.0f;
            cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
            fn(d_input, d_result, n, args.block_size);
        }
        for (int i = 0; i < args.iters; ++i) {
            float zero = 0.0f;
            cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
            times_ms.push_back(fn(d_input, d_result, n, args.block_size).kernel_time_ms);
        }

        bytes_moved = static_cast<double>(n) * sizeof(float);
        flops = static_cast<double>(n);
        cudaFree(d_input); cudaFree(d_result);

    } else if (args.kernel == "matmul_naive" || args.kernel == "matmul_tiled") {
        auto fn = (args.kernel == "matmul_naive") ? thermal_kernels::matmul_naive : thermal_kernels::matmul_tiled;
        correct = verifyMatmul(fn);
        if (!correct) { printf("{\"kernel\":\"%s\",\"correct\":false}\n", args.kernel.c_str()); return 1; }

        int n = args.n;
        std::vector<float> a(n * n, 1.0f), b(n * n, 1.0f);
        float *d_a, *d_b, *d_c;
        checkCuda(cudaMalloc(&d_a, n * n * sizeof(float)), "malloc a");
        checkCuda(cudaMalloc(&d_b, n * n * sizeof(float)), "malloc b");
        checkCuda(cudaMalloc(&d_c, n * n * sizeof(float)), "malloc c");
        cudaMemcpy(d_a, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

        for (int i = 0; i < args.warmup; ++i) fn(d_a, d_b, d_c, n);
        for (int i = 0; i < args.iters; ++i) times_ms.push_back(fn(d_a, d_b, d_c, n).kernel_time_ms);

        bytes_moved = 3.0 * n * n * sizeof(float);
        flops = 2.0 * n * n * n;
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    } else {
        fprintf(stderr, "unknown kernel '%s'\n", args.kernel.c_str());
        return 1;
    }

    printf("{\n");
    printf("  \"kernel\": \"%s\",\n", args.kernel.c_str());
    printf("  \"device\": \"%s\",\n", prop.name);
    printf("  \"n\": %d,\n", args.n);
    printf("  \"block_size\": %d,\n", args.block_size);
    printf("  \"correct\": true,\n");
    printf("  \"bytes_moved\": %.0f,\n", bytes_moved);
    printf("  \"flops\": %.0f,\n", flops);
    printf("  \"iteration_times_ms\": [");
    for (size_t i = 0; i < times_ms.size(); ++i) {
        printf("%s%.6f", (i == 0 ? "" : ", "), times_ms[i]);
    }
    printf("]\n");
    printf("}\n");

    return 0;
}
