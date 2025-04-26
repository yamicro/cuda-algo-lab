//
// Created by yami on 25-4-26.
//
#include <cuda_runtime.h>

__global__ void mat_transpose_f32_col2row_kernel(
  float *x, float *y, const int row, const int col) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_row = global_idx / col;
    const int global_col = global_idx % col;
    if (global_idx < row * col) {
        y[global_col * row + global_row] = x[global_idx];
    }
}