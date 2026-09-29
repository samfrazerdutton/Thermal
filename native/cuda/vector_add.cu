#include "kernels.h"

#include <cuda_runtime.h>

namespace {

__global__ void vectorAddNaiveKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vectorAddGridStrideKernel(const float* a, const float* b, float* c, int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

}  // namespace

namespace thermal_kernels {

LaunchResult vector_add_naive(const float* d_a, const float* d_b, float* d_c, int n, int block_size) {
    int grid_size = (n + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAddNaiveKernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {ms};
}

LaunchResult vector_add_grid_stride(const float* d_a, const float* d_b, float* d_c, int n, int block_size) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    // A fixed grid sized to keep every SM saturated, regardless of n.
    int grid_size = prop.multiProcessorCount * 32;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAddGridStrideKernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {ms};
}

}  // namespace thermal_kernels
