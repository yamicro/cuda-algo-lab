//
// Created by yami on 25-4-24.
//
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "util/half_exp.h"

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__device__ __forceinline__ float swish(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ half swish_half(half x) {
  return __hmul(x, __hdiv(__float2half(1.0f),
                          __hadd(__float2half(1.0f), half_exp(__hneg(x)))));
}

__global__ void swish_f32_kernel(float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = swish(x[idx]);
}

__global__ void swish_f16x8_pack_kernel(half *x, half *y, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half pack_x[8], pack_y[8];
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
  for (int i = 0; i < 8; i++) {
    pack_y[i] = swish_half(pack_x[i]);
  }
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}