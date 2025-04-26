//
// Created by yami on 25-4-26.
//
#include <cuda_runtime.h>

__global__ void embedding_f32_kernel(const int *idx, float *weight, float *output, int n, int emb_size)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = bx * blockDim.x + tx;
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
}