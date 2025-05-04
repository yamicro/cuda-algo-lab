//
// Created by yami on 25-4-26.
//
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SIZE 256
#define WARP_SIZE_S 16
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
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

__global__ void mat_transpose_f32x4_shared_col2row2d_kernel(
    float *x, float *y,
    const int row,      // 输入矩阵的行数
    const int col       // 输入矩阵的列数
) {
  // 1. 全局索引（在原矩阵 x 中要读的位置）
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  // 2. 局部索引（在 block 内部 shared memory tile 的位置）
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;

  // 3. 申明 shared memory tile（二维），宽度是 warp 大小×4
  __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4];

  // 4. 边界检查——保证我们加载的 “一个 float4”（4 个 float）不越界
  //    global_x * 4 + 3 < col + 3  等价于 global_x*4+3 < col
  //    global_y < row
  if (global_x * 4 + 3 < col + 3 && global_y < row) {
    // 5. 以 float4 为单位，共享内存的协同读写可以做到更高的带宽利用
    //    reinterpret_cast<float4*>(x) 让我们可以一次性读取 4 个 float
    float4 x_val = reinterpret_cast<float4 *>(x)[ global_y * (col/4) + global_x ];
    // 6. 把这 4 个 float 存到 shared memory tile 中
    //    FLOAT4(tile[local_y][local_x * 4]) = FLOAT4(x_val);
    //    宏展开后等价于下面两步：
    //    tile[ local_y ][ local_x*4 + 0 ] = x_val.x;
    //    tile[ local_y ][ local_x*4 + 1 ] = x_val.y;
    //    tile[ local_y ][ local_x*4 + 2 ] = x_val.z;
    //    tile[ local_y ][ local_x*4 + 3 ] = x_val.w;
    FLOAT4(tile[local_y][local_x * 4]) = FLOAT4(x_val);

    __syncthreads();  // 确保整个 tile 已经写完，才能读

    // 7. 从 shared memory tile 里读出要写到 y 的 4 个元素，
    //    并同时做行列交换（转置）和地址重排
    float4 smem_val;
    // STRIDE = WARP_SIZE_S / 4，用来把 tile 划分成 4 段
    constexpr int STRIDE = WARP_SIZE_S / 4;
    // 读 tile 时，将行、列重新映射：
    //  - (local_y % STRIDE)*4 + offset: 选出哪一段
    //  - local_x*4 + local_y/STRIDE: 选出该段里的哪一列
    smem_val.x = tile[(local_y % STRIDE) * 4]
                     [ local_x*4 + local_y/STRIDE ];
    smem_val.y = tile[(local_y % STRIDE) * 4 + 1]
                     [ local_x*4 + local_y/STRIDE ];
    smem_val.z = tile[(local_y % STRIDE) * 4 + 2]
                     [ local_x*4 + local_y/STRIDE ];
    smem_val.w = tile[(local_y % STRIDE) * 4 + 3]
                     [ local_x*4 + local_y/STRIDE ];

    // 8. 计算写回 y 矩阵的全局坐标
    //    假设原来 x 是 row×col，现在 y 是 col×row
    const int bid_y = blockIdx.y * blockDim.y;       // block 在 y 方向的顶点行数
    const int out_y = global_x * 4 + local_y / STRIDE;
    const int out_x = (local_y % STRIDE) * 4 + bid_y;

    // 再次用 float4 写：一次写 4 个 float
    reinterpret_cast<float4 *>(y)[ (out_y * row + out_x) / 4 ]
      = FLOAT4(smem_val);
  }
}