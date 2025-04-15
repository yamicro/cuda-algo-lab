//
// Created by yami on 25-4-15.
//
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_EXP_F32  88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f

__global__ void sigmoid_kernel(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = x[idx];
    v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);
    y[idx] = 1.0f / (1.0f + expf(-v));
  }
}