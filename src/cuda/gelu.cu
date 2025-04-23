//
// Created by yami on 25-4-23.
//
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_kernel(const float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = gelu_approx(x[idx]);
    }
}