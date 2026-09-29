#pragma once

// Host-callable launchers for the THERMAL kernel lab. Each function times only
// the kernel launch itself (via CUDA events) -- allocation and host<->device
// transfer are the caller's responsibility and are timed separately by
// bench_main.cu, so a benchmark result never conflates transfer cost with
// compute cost.

namespace thermal_kernels {

struct LaunchResult {
    float kernel_time_ms;
};

// c[i] = a[i] + b[i]. One thread per element, no grid-stride loop -- breaks
// down (silently under-computes) if n > gridDim.x * blockDim.x, which the
// caller must guarantee.
LaunchResult vector_add_naive(const float* d_a, const float* d_b, float* d_c, int n, int block_size);

// Same operation, using a grid-stride loop so a fixed, hardware-sized grid
// can process an arbitrarily large n -- the standard "optimized" pattern for
// this kernel on hardware where memory bandwidth, not launch overhead, is
// the ceiling.
LaunchResult vector_add_grid_stride(const float* d_a, const float* d_b, float* d_c, int n, int block_size);

// result must point to a single zero-initialized device float. Every thread
// issues its own atomicAdd -- included specifically to demonstrate atomic
// contention as a bottleneck, not as a recommended pattern.
LaunchResult reduce_naive_atomic(const float* d_input, float* d_result, int n, int block_size);

// result must point to a single zero-initialized device float. Each block
// reduces its chunk in shared memory (tree reduction) and issues exactly one
// atomicAdd per block.
LaunchResult reduce_shared_memory(const float* d_input, float* d_result, int n, int block_size);

// C = A * B, all N x N, row-major. One thread per output element, reading
// operands directly from global memory on every step -- deliberately
// bandwidth-bound to contrast with the tiled version below.
LaunchResult matmul_naive(const float* d_a, const float* d_b, float* d_c, int n);

// Same operation using shared-memory tiling (TILE_WIDTH x TILE_WIDTH tiles)
// to cut global memory traffic by reusing each loaded element TILE_WIDTH
// times from shared memory.
LaunchResult matmul_tiled(const float* d_a, const float* d_b, float* d_c, int n);

constexpr int kMatmulTileWidth = 16;

}  // namespace thermal_kernels
