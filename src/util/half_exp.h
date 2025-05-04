//
// Created by yami on 25-5-4.
//
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#ifndef HALF_EXP_H
#define HALF_EXP_H
__device__ __forceinline__ __half half_exp(const __half h) {
    return __float2half_rn(expf(__half2float(h)));
}
#endif //HALF_EXP_H
