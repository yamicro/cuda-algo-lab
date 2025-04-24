//
// Created by yami on 25-4-24.
//
#include <cuda_runtime.h>

__device__ __forceinline__ float swish(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void swish_f32_kernel(float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = swish(x[idx]);
}