//
// Created by yami on 25-4-26.
//
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void embedding_f32_kernel(const int *idx, float *weight, float *output, int n, int emb_size)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = bx * blockDim.x + tx;
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
}

__global__ void embedding_f16x8_pack_kernel(const int *idx, half *weight,
                                            half *output, int n, int emb_size) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = bx * blockDim.x + tx;
    int offset = idx[bx] * emb_size;
    LDST128BITS(output[bx * emb_size + 8 * tx]) =
        LDST128BITS(weight[offset + 8 * tx]);
}