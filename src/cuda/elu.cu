//
// Created by yami on 25-4-22.
//
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>


#define ALPHA 1.0f
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__device__ __forceinline__ __half half_exp(const __half h) {
    return __float2half_rn(expf(__half2float(h)));
}

__device__ __forceinline__ float elu(float x) {
    return x > 0.f ? x : ALPHA * (expf(x) - 1.f);
}
__device__ __forceinline__ __half elu_half(__half x) {
    const __half one   = __float2half(1.f);
    const __half alpha = __float2half(ALPHA);

    return __hgt(x, __float2half(0.f))
           ? x
           : __hmul(alpha, __hsub(half_exp(x), one));
}

__global__ void elu_f32_kernel(float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = elu(x[idx]);
}

__global__ void elu_f16x8_pack_kernel(half *x, float *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half pack_x[8], pack_y[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = idx + i;
        pack_y[i] = elu_half(pack_x[i]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = idx + i;
        if (idx < N) y[idx] = pack_y[i];
    }
}