//
// Created by yami on 25-4-23.
//
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include "util/half_exp.h"

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)
#define SQRT_2_PI M_SQRT2 *M_2_SQRTPI * 0.5f
#define HALF_1 __float2half(1.0f)
#define HALF_2 __float2half(2.0f)
#define HALF_DIV2 __float2half(0.5f)
#define HALF_SQRT_2_PI                                                         \
  __float2half(M_SQRT2) * __float2half(M_2_SQRTPI) * HALF_DIV2
#define HALF_V_APP __float2half(0.044715f)
#define HALF_GELU_OPS gelu_tanh_approximate

__inline__ __device__ half gelu_tanh_approximate(half x) {
  half x_cube = x * x * x;
  // compute mid value : inner = 0.7978845608 * (x + 0.044715 * x * x * x)
  half inner = HALF_SQRT_2_PI * (x + HALF_V_APP * x_cube);
  // compute tanh
  return HALF_DIV2 * x *
         (HALF_1 +
          ((half_exp(inner * HALF_2) - HALF_1) / (half_exp(inner * HALF_2) + HALF_1)));
}

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

__global__ void gelu_f16x8_pack_kernel(half *x, half *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

  // temporary register(memory), .local space in ptx, addressable
  half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); // load 128 bits

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    half v = __hmin(__hmax(pack_x[i], MIN_EXP_F16), MAX_EXP_F16);
    pack_y[i] = HALF_GELU_OPS(v);
  }
  // reinterpret as float4 and store 128 bits in 1 memory issue.
  if ((idx + 7) < N) {
    LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
  }
}