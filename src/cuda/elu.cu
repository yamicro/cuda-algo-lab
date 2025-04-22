//
// Created by yami on 25-4-22.
//
#include <cuda_runtime.h>
#include <algorithm>
#include <stdio.h>

#define ALPHA 1.0f

__device__ __forceinline__ float elu(float x) {
    return x > 0.f ? x : ALPHA * (expf(x) - 1.f);
}

__global__ void elu_f32_kernel(float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = elu(x[idx]);
}