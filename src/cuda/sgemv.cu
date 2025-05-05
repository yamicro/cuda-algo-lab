#include <cuda_runtime.h>

#define WARP_SIZE 32
#define FLOAT4(x) (*reinterpret_cast<float4*>(&(x)))

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32_sgemv(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 版本1：K 为 32 倍数
__global__ void sgemv_k32_f32_kernel(float *a, float *x, float *y, int M, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int lane = tx % WARP_SIZE;
    int m = bx * blockDim.y + ty;

    if (m < M) {
        float sum = 0.0f;
        int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k = w * WARP_SIZE + lane;
            sum += a[m * K + k] * x[k];
        }
        sum = warp_reduce_sum_f32_sgemv<WARP_SIZE>(sum);
        if (lane == 0)
            y[m] = sum;
    }
}

// 版本2：K 为 128 倍数，float4 向量化
__global__ void sgemv_k128_f32x4_kernel(float *a, float *x, float *y, int M, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int lane = tx % WARP_SIZE;
    int m = blockDim.y * bx + ty;

    if (m < M) {
        float sum = 0.0f;
        int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            int k = (w * WARP_SIZE + lane) * 4;
            float4 reg_x = FLOAT4(x[k]);
            float4 reg_a = FLOAT4(a[m * K + k]);
            sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y + reg_a.z * reg_x.z + reg_a.w * reg_x.w);
        }
        sum = warp_reduce_sum_f32_sgemv<WARP_SIZE>(sum);
        if (lane == 0)
            y[m] = sum;
    }
}
