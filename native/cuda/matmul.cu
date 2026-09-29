#include "kernels.h"

#include <cuda_runtime.h>

namespace {

__global__ void matmulNaiveKernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void matmulTiledKernel(const float* a, const float* b, float* c, int n) {
    constexpr int TILE = thermal_kernels::kMatmulTileWidth;
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? a[row * n + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? b[bRow * n + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

}  // namespace

namespace thermal_kernels {

LaunchResult matmul_naive(const float* d_a, const float* d_b, float* d_c, int n) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmulNaiveKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {ms};
}

LaunchResult matmul_tiled(const float* d_a, const float* d_b, float* d_c, int n) {
    dim3 block(kMatmulTileWidth, kMatmulTileWidth);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmulTiledKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {ms};
}

}  // namespace thermal_kernels
