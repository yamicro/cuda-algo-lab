//
// Created by yami on 25-4-28.
//
#include <cuda_runtime.h>

#define WARP_SIZE 32

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Block All Reduce Sum
// grid(N/256), block(256)
// a: Nx1, y=sum(a)
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    // keep the data in register is enough for warp operaion.
    float sum = (idx < N) ? a[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // perform warp sync reduce.
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    // warp leaders store the data to shared memory.
    if (lane == 0)
        reduce_smem[warp] = sum;
    __syncthreads(); // make sure the data is in shared memory.
    // the first warp compute the final sum.
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}