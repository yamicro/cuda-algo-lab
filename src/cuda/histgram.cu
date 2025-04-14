#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>


__global__ void histogram_kernel(const int* a, int* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&(y[a[idx]]), 1);
    }
}
