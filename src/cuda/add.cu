#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

__global__ void add_fp16_pack_kernel(half* a,
                                     half* b,
                                     float*       c,
                                     int          n) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (base >= n) return;

    half pack_a[8];
    half pack_b[8];

    LDST128BITS(pack_a[0]) = LDST128BITS(a[base]);
    LDST128BITS(pack_b[0]) = LDST128BITS(b[base]);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = base + i;
            half  sum_h = __hadd(pack_a[i], pack_b[i]);
            c[idx]      = __half2float(sum_h);
    }
}