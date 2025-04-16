//
// Created by yami on 25-4-16.
//

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { y[idx] = fmaxf(0.0f, x[idx]); }
}