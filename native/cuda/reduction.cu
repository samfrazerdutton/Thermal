#include "kernels.h"

#include <cuda_runtime.h>

namespace {

__global__ void reduceNaiveAtomicKernel(const float* input, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(result, input[i]);
    }
}

__global__ void reduceSharedMemoryKernel(const float* input, float* result, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

}  // namespace

namespace thermal_kernels {

LaunchResult reduce_naive_atomic(const float* d_input, float* d_result, int n, int block_size) {
    int grid_size = (n + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduceNaiveAtomicKernel<<<grid_size, block_size>>>(d_input, d_result, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {ms};
}

LaunchResult reduce_shared_memory(const float* d_input, float* d_result, int n, int block_size) {
    int grid_size = (n + block_size - 1) / block_size;
    size_t shared_bytes = block_size * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduceSharedMemoryKernel<<<grid_size, block_size, shared_bytes>>>(d_input, d_result, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {ms};
}

}  // namespace thermal_kernels
