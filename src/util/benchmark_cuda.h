//
// Created by yami on 25-4-16.
//

#ifndef BENCHMARK_CUDA_H
#define BENCHMARK_CUDA_H

#pragma once
#include <cuda_runtime.h>
#include <functional>

float benchmark_kernel(const std::function<void()>& kernel_launcher,
                       int warmup_iters = 1, int timing_iters = 1) {
    // 预热：不计时
    for (int i = 0; i < warmup_iters; ++i) {
        kernel_launcher();
    }
    cudaDeviceSynchronize(); // 确保预热完成

    // 正式开始计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < timing_iters; ++i) {
        kernel_launcher();
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / timing_iters;  // 返回单次平均时间
}



#endif //BENCHMARK_CUDA_H
