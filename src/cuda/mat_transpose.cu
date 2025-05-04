//
// Created by yami on 25-4-26.
//
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void mat_transpose_f32_col2row_kernel(
  float *x, float *y, const int row, const int col) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_row = global_idx / col;
    const int global_col = global_idx % col;
    if (global_idx < row * col) {
        y[global_col * row + global_row] = x[global_idx];
    }
}
