#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "util/benchmark_cuda.h"
#include "cuda/add.cu"
#include "cuda/histgram.cu"
#include "cuda/sigmod.cu"
#include "cuda/relu.cu"
#include "cuda/elu.cu"
#include "cuda/gelu.cu"
#include "cuda/swish.cu"
#include "cuda/embedding.cu"
#include "cuda/mat_transpose.cu"
#include "cuda/warp_reduce_sum.cu"

#ifndef PYBIND11_HALF_T_DEFINED
#define PYBIND11_HALF_T_DEFINED
namespace pybind11 { using half_t = uint16_t; }
#endif




float add_cuda(pybind11::array_t<float> a, pybind11::array_t<float> b, pybind11::array_t<float> c) {
    auto buf_a = a.unchecked<1>();
    auto buf_b = b.unchecked<1>();
    auto buf_c = c.mutable_unchecked<1>();
    int n = buf_a.size();

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));

    cudaMemcpy(d_a, buf_a.data(0), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, buf_b.data(0), n * sizeof(float), cudaMemcpyHostToDevice);

    float elapsed = benchmark_kernel([&]() {
        add_kernel<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);
    }, 3, 10);


    cudaMemcpy(buf_c.mutable_data(0), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return elapsed;
}

float add_fp16_pack_cuda_trans_after(pybind11::array_t<float> a, pybind11::array_t<float> b, pybind11::array_t<float> c) {
    auto buf_a = a.unchecked<1>();
    auto buf_b = b.unchecked<1>();
    auto buf_c = c.mutable_unchecked<1>();
    int n = buf_a.size();

    int N = buf_a.size();
    if (N % 8 != 0) throw std::runtime_error("Input size must be divisible by 8.");

    float* h_a = const_cast<float*>(buf_a.data(0));
    half* d_a;

    float* h_b = const_cast<float*>(buf_b.data(0));
    half* d_b;

    cudaMalloc(&d_a, N * sizeof(half));
    cudaMalloc(&d_b, N * sizeof(half));

    cudaMemcpy(d_a, buf_a.data(0), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, buf_b.data(0), n * sizeof(float), cudaMemcpyHostToDevice);
    float* d_c;
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMemset(d_c, 0, N * sizeof(float));

    float elapsed = benchmark_kernel([&]() {
        add_fp16_pack_kernel<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);
    }, 3, 10);


    cudaMemcpy(buf_c.mutable_data(0), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return elapsed;
}

float add_fp16_pack_cuda(pybind11::array_t<pybind11::half_t,
                                                  pybind11::array::c_style |
                                                  pybind11::array::forcecast> a,
                                      pybind11::array_t<pybind11::half_t,
                                                  pybind11::array::c_style |
                                                  pybind11::array::forcecast> b) {

    const int N = static_cast<int>(a.size());

    auto out = pybind11::array_t<float>(N);
    float* h_out = out.mutable_data();

    const __half* h_a = reinterpret_cast<const __half*>(a.data());
    const __half* h_b = reinterpret_cast<const __half*>(b.data());

    __half* d_a;  __half* d_b;
    float * d_c;
    cudaMalloc(&d_a, N * sizeof(__half));
    cudaMalloc(&d_b, N * sizeof(__half));
    cudaMalloc(&d_c, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(__half), cudaMemcpyHostToDevice);

    constexpr int THREADS = 128;
    constexpr int VEC = 8;
    int elems_per_block = THREADS * VEC;
    int grid = (N + elems_per_block - 1) / elems_per_block;

    float elapsed = benchmark_kernel([&]() {
        add_fp16_pack_kernel<<<grid, THREADS>>>(d_a, d_b, d_c, N);
    }, 3, 10);

    cudaMemcpy(h_out, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return elapsed;
}


float histogram_cuda(pybind11::array_t<int> a, pybind11::array_t<int> y) {
    auto buf_a = a.unchecked<1>();
    auto buf_y = y.mutable_unchecked<1>();
    int N = buf_a.size();

    int *d_a, *d_y;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_y, y.size() * sizeof(int));

    cudaMemcpy(d_a, buf_a.data(0), N * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_y, buf_y.data(0), buf_y.size(0) * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float elapsed = benchmark_kernel([&]() {
        histogram_kernel<<<blocks, threads>>>(d_a, d_y, N);
    }, 3, 10);


    cudaMemcpy(buf_y.mutable_data(0), d_y, buf_y.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_y);

    return elapsed;
}

float sigmoid_cuda(pybind11::array_t<float> x, pybind11::array_t<float> y) {
    auto buf_x = x.unchecked<1>();
    auto buf_y = y.mutable_unchecked<1>();
    int N = buf_x.size();

    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    cudaMemcpy(d_x, buf_x.data(0), N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float elapsed = benchmark_kernel([&]() {
        sigmoid_kernel<<<blocks, threads>>>(d_x, d_y, N);
    }, 3, 10);

    cudaMemcpy(buf_y.mutable_data(0), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);

    return elapsed;
}

float sigmoid_cuda_fp16_pack(pybind11::array_t<pybind11::half_t,
                                         pybind11::array::c_style |
                                         pybind11::array::forcecast> x,
                             pybind11::array_t<pybind11::half_t,
                                         pybind11::array::c_style |
                                         pybind11::array::forcecast> y,
                             int M) {

    const int N = static_cast<int>(x.size());

    const __half* h_x = reinterpret_cast<const __half*>(x.data());
    __half*       h_y = reinterpret_cast<__half*>(y.mutable_data());

    __half *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, N * sizeof(__half));
    cudaMalloc(&d_y, N * sizeof(__half));

    cudaMemcpy(d_x, h_x, N * sizeof(__half), cudaMemcpyHostToDevice);

    constexpr int THREADS = 256;
    int blocks = (N + THREADS - 1) / THREADS;

    float elapsed = benchmark_kernel([&](){
        sigmoid_f16x8_pack_kernel<<<blocks, THREADS>>>(d_x, d_y, N);
    }, 3, 10);

    cudaMemcpy(h_y, d_y, N * sizeof(__half), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    return elapsed;
}

float relu_cuda(pybind11::array_t<float> x, pybind11::array_t<float> y) {
    auto buf_x = x.unchecked<1>();
    auto buf_y = y.mutable_unchecked<1>();
    int N = buf_x.size();

    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    cudaMemcpy(d_x, buf_x.data(0), N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float elapsed = benchmark_kernel([&]() {
        relu_kernel<<<blocks, threads>>>(d_x, d_y, N);
    },3, 10);

    cudaMemcpy(buf_y.mutable_data(0), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);

    return elapsed;
}

float relu_cuda_fp16_pack(pybind11::array_t<pybind11::half_t,
                                         pybind11::array::c_style |
                                         pybind11::array::forcecast> x,
                             pybind11::array_t<pybind11::half_t,
                                         pybind11::array::c_style |
                                         pybind11::array::forcecast> y) {

    const int N = static_cast<int>(x.size());

    const __half* h_x = reinterpret_cast<const __half*>(x.data());
    __half*       h_y = reinterpret_cast<__half*>(y.mutable_data());

    __half *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, N * sizeof(__half));
    cudaMalloc(&d_y, N * sizeof(__half));

    cudaMemcpy(d_x, h_x, N * sizeof(__half), cudaMemcpyHostToDevice);

    constexpr int THREADS = 256;
    int blocks = (N + THREADS - 1) / THREADS;

    float elapsed = benchmark_kernel([&](){
        relu_f16x8_pack_kernel<<<blocks, THREADS>>>(d_x, d_y, N);
    }, 3, 10);

    cudaMemcpy(h_y, d_y, N * sizeof(__half), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    return elapsed;
}

float elu_cuda(pybind11::array_t<float> x, pybind11::array_t<float> y) {
    auto buf_x = x.unchecked<1>();
    auto buf_y = y.mutable_unchecked<1>();
    int N = buf_x.size();

    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    cudaMemcpy(d_x, buf_x.data(0), N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

	float elapsed = benchmark_kernel([&]() {
        elu_f32_kernel<<<blocks, threads>>>(d_x, d_y, N);
    },3, 10);

    cudaMemcpy(buf_y.mutable_data(0), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    return elapsed;
}

float elu_cuda_fp16_pack(pybind11::array_t<pybind11::half_t,
                                         pybind11::array::c_style |
                                         pybind11::array::forcecast> x) {

    const int N = static_cast<int>(x.size());

    auto y = pybind11::array_t<float>(N);
    float* h_out = y.mutable_data();

    const half* h_in = reinterpret_cast<const half*>(x.data());
    half*  d_in  = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in,  N * sizeof(half));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(half), cudaMemcpyHostToDevice);

    constexpr int THREADS = 128;
    constexpr int VEC = 8;
    int elems_per_block = THREADS * VEC;
    int grid = (N + elems_per_block - 1) / elems_per_block;

    auto t0 = std::chrono::high_resolution_clock::now();
    float elapsed = benchmark_kernel([&]() {
        elu_f16x8_pack_kernel<<<grid, THREADS>>>(d_in, d_out, N);
    },3, 10);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    /* ---- D→H ---- */
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    return elapsed;
}

float gelu_cuda(pybind11::array_t<float> x, pybind11::array_t<float> y) {
    auto buf_x = x.unchecked<1>();
    auto buf_y = y.mutable_unchecked<1>();
    int N = buf_x.size();

    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    cudaMemcpy(d_x, buf_x.data(0), N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float elapsed = benchmark_kernel([&]() {
    	gelu_kernel<<<blocks, threads>>>(d_x, d_y, N);
    },3, 10);

    cudaMemcpy(buf_y.mutable_data(0), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    return elapsed;
}

float gelu_cuda_fp16_pack(pybind11::array_t<pybind11::half_t,
                                      pybind11::array::c_style |
                                      pybind11::array::forcecast> x,
                          pybind11::array_t<pybind11::half_t,
                                      pybind11::array::c_style |
                                      pybind11::array::forcecast> y) {
    const int N = static_cast<int>(x.size());

    const __half* h_x = reinterpret_cast<const __half*>(x.data());
    __half*       h_y = reinterpret_cast<__half*>(y.mutable_data());

    __half *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, N * sizeof(__half));
    cudaMalloc(&d_y, N * sizeof(__half));

    cudaMemcpy(d_x, h_x, N * sizeof(__half), cudaMemcpyHostToDevice);

    constexpr int THREADS = 256;
    int blocks = (N + THREADS - 1) / THREADS;

    float elapsed = benchmark_kernel([&]() {
        gelu_f16x8_pack_kernel<<<blocks, THREADS>>>(d_x, d_y, N);
    }, 3, 10);

    cudaMemcpy(h_y, d_y, N * sizeof(__half), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    return elapsed;
}

float swish_cuda(pybind11::array_t<float> x, pybind11::array_t<float> y) {
    auto buf_x = x.unchecked<1>();
    auto buf_y = y.mutable_unchecked<1>();
    int N = buf_x.size();

    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    cudaMemcpy(d_x, buf_x.data(0), N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float elapsed = benchmark_kernel([&]() {
    	swish_f32_kernel<<<blocks, threads>>>(d_x, d_y, N);
    },3, 10);

    cudaMemcpy(buf_y.mutable_data(0), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    return elapsed;
}

float swish_cuda_fp16_pack(pybind11::array_t<pybind11::half_t,
                                      pybind11::array::c_style |
                                      pybind11::array::forcecast> x,
                          pybind11::array_t<pybind11::half_t,
                                      pybind11::array::c_style |
                                      pybind11::array::forcecast> y) {
    const int N = static_cast<int>(x.size());

    const __half* h_x = reinterpret_cast<const __half*>(x.data());
    __half*       h_y = reinterpret_cast<__half*>(y.mutable_data());

    __half *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, N * sizeof(__half));
    cudaMalloc(&d_y, N * sizeof(__half));

    cudaMemcpy(d_x, h_x, N * sizeof(__half), cudaMemcpyHostToDevice);

    constexpr int THREADS = 256;
    int blocks = (N + THREADS - 1) / THREADS;

    float elapsed = benchmark_kernel([&]() {
        swish_f16x8_pack_kernel<<<blocks, THREADS>>>(d_x, d_y, N);
    }, 3, 10);

    /* ---------- D→H ---------- */
    cudaMemcpy(h_y, d_y, N * sizeof(__half), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    return elapsed;
}

float embedding_cuda(pybind11::array_t<int> input,pybind11::array_t<float> weights, pybind11::array_t<float> output) {
    auto buf_input = input.unchecked<1>();
    auto buf_weights = weights.unchecked<2>();
    auto buf_output_info = output.request();
    float *h_output = static_cast<float *>(buf_output_info.ptr);

    int N = buf_input.size();
    int D = buf_weights.shape(1);

    int *d_indices;
    float *d_weights, *d_output;
    cudaMalloc(&d_indices, N * sizeof(int));
    cudaMalloc(&d_weights, buf_weights.shape(0) * D * sizeof(float));
    cudaMalloc(&d_output, N * D * sizeof(float));

    cudaMemcpy(d_indices, buf_input.data(0), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, buf_weights.data(0, 0), buf_weights.shape(0) * D * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float elapsed = benchmark_kernel([&]() {
        embedding_f32_kernel<<<blocks, threads>>>(d_indices, d_weights, d_output, N, D);
    }, 3, 10);

    cudaMemcpy(h_output, d_output, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_indices);
    cudaFree(d_weights);
    cudaFree(d_output);

    return elapsed;
}

float embedding_cuda_fp16_pack(pybind11::array_t<int,
                                      pybind11::array::c_style |
                                      pybind11::array::forcecast>      input,
                          pybind11::array_t<pybind11::half_t,
                                      pybind11::array::c_style |
                                      pybind11::array::forcecast>      weights,
                          pybind11::array_t<pybind11::half_t,
                                      pybind11::array::c_style |
                                      pybind11::array::forcecast>      output) {
    auto in  = input.unchecked<1>();
    auto w   = weights.unchecked<2>();
    auto out_info = output.request();
    __half* h_out = reinterpret_cast<__half*>(out_info.ptr);

    const int N = static_cast<int>(in.size());
    const int D = static_cast<int>(w.shape(1));
    const int V = static_cast<int>(w.shape(0));
    if (output.size() != static_cast<size_t>(N * D))
        throw std::runtime_error("output shape 不匹配");

    int*   d_idx     = nullptr;
    __half* d_w      = nullptr;
    __half* d_out    = nullptr;
    cudaMalloc(&d_idx,  N * sizeof(int));
    cudaMalloc(&d_w,   V * D * sizeof(__half));
    cudaMalloc(&d_out, N * D * sizeof(__half));

    const int*   h_idx = in.data(0);
    const __half* h_w  = reinterpret_cast<const __half*>(w.data(0,0));

    cudaMemcpy(d_idx, h_idx, N * sizeof(int),            cudaMemcpyHostToDevice);
    cudaMemcpy(d_w,   h_w,  V * D * sizeof(__half),      cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    float elapsed = benchmark_kernel([&](){
        embedding_f16x8_pack_kernel<<<blocks, threads>>>(d_idx, d_w, d_out, N, D);
    }, 3, 10);

    cudaMemcpy(h_out, d_out, N * D * sizeof(__half),     cudaMemcpyDeviceToHost);

    cudaFree(d_idx);
    cudaFree(d_w);
    cudaFree(d_out);

    return elapsed;
}

float mat_transpose_cuda(pybind11::array_t<int> input, pybind11::array_t<float> output) {
    auto buf_input = input.unchecked<2>();
    auto buf_output = output.mutable_unchecked<2>();

    int N = buf_input.shape(0);
    int D = buf_input.shape(1);

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, N * D * sizeof(float));
    cudaMalloc(&d_output, N * D * sizeof(float));

    cudaMemcpy(d_input, buf_input.data(0, 0), N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, buf_input.data(0, 0), N * D * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float elapsed = benchmark_kernel([&]() {
        mat_transpose_f32_col2row_kernel<<<blocks, threads>>>(d_input, d_output, N, D);
    }, 3, 10);

    cudaMemcpy(buf_output.mutable_data(0,0), d_output, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return elapsed;
}


float warp_reduce_sum_cuda(pybind11::array_t<float> input, pybind11::array_t<float> output) {
    auto buf_in = input.unchecked<1>();
    auto buf_out = output.mutable_unchecked<1>();

    int N = buf_in.size();

    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, (N / 32) * sizeof(float));

    cudaMemcpy(d_x, buf_in.data(0), N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float elapsed = benchmark_kernel([&]() {
    	block_all_reduce_sum_f32_f32_kernel<<<blocks, threads>>>(d_x, d_y, N);
    }, 3, 10);


    cudaMemcpy(buf_out.mutable_data(0), d_y, (N / 32) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    return elapsed;
}

float warp_reduce_fp16_cuda_trans_after(pybind11::array_t<float> input, pybind11::array_t<float> output) {
    auto buf_in = input.unchecked<1>();
    auto buf_out = output.mutable_unchecked<1>();

    int N = buf_in.size();
    if (N % 8 != 0) throw std::runtime_error("Input size must be divisible by 8.");

    float* h_input = const_cast<float*>(buf_in.data(0));
    half* d_input;
    float* d_output;

    cudaMalloc(&d_input, N * sizeof(half));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemset(d_output, 0, sizeof(float));

    // convert to half
    std::vector<half> h_half(N);
    for (int i = 0; i < N; ++i)
        h_half[i] = __float2half(h_input[i]);

    cudaMemcpy(d_input, h_half.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float elapsed = benchmark_kernel([&]() {
    	block_all_reduce_sum_f16x8_pack_f16_kernel<<<blocks, threads>>>(d_input, d_output, N);
    }, 3, 10);

    cudaMemcpy(buf_out.mutable_data(0), d_output, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    return elapsed;
}

float warp_reduce_fp16_cuda(pybind11::array_t<pybind11::half_t,
                                       pybind11::array::c_style |
                                       pybind11::array::forcecast> input,
                            pybind11::array_t<float,
                                       pybind11::array::c_style | pybind11::array::forcecast> output) {

    const int N = static_cast<int>(input.size());

    const __half* h_in = reinterpret_cast<const __half*>(input.data());
    float*        h_out = output.mutable_data();

    __half* d_in  = nullptr;
    float*  d_out = nullptr;
    cudaMalloc(&d_in,  N * sizeof(__half));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(__half), cudaMemcpyHostToDevice);

    constexpr int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float elapsed = benchmark_kernel([&]() {
    	block_all_reduce_sum_f16x8_pack_f16_kernel<<<blocks, threads>>>(d_in, d_out, N);
    }, 3, 10);
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    return elapsed;
}



PYBIND11_MODULE(binding, m) {
    m.def("add_cuda", &add_cuda, "CUDA add two arrays");
    m.def("add_fp16_pack_cuda", &add_fp16_pack_cuda, "CUDA add two arrays in fp16");
    m.def("histogram_cuda", &histogram_cuda, "CUDA histogram");
    m.def("sigmoid_cuda", &sigmoid_cuda, "CUDA sigmoid");
    m.def("sigmoid_cuda_fp16_pack", &sigmoid_cuda_fp16_pack, "CUDA sigmoid fp16");
    m.def("relu_cuda", &relu_cuda, "CUDA relu");
    m.def("relu_cuda_fp16_pack", &relu_cuda_fp16_pack, "CUDA relu fp16");
    m.def("elu_cuda", &elu_cuda, "CUDA ELU");
    m.def("elu_cuda_fp16_pack", &elu_cuda_fp16_pack, "CUDA elu_cuda_fp16_pack");
    m.def("gelu_cuda", &gelu_cuda, "CUDA GELU");
    m.def("gelu_cuda_fp16_pack", &gelu_cuda_fp16_pack, "CUDA GELU gelu_cuda_fp16_pack");
    m.def("swish_cuda", &swish_cuda, "CUDA SWISH");
    m.def("swish_cuda_fp16_pack", &swish_cuda_fp16_pack, "CUDA fp16 swish");
    m.def("embedding_cuda", &embedding_cuda, "CUDA embedding");
    m.def("embedding_cuda_fp16_pack", &embedding_cuda_fp16_pack, "CUDA embedding_cuda_fp16_pack");
    m.def("mat_transpose_cuda", &mat_transpose_cuda, "CUDA mat_transpose transpose");
	m.def("warp_reduce_sum_cuda", &warp_reduce_sum_cuda, "CUDA warp reduce sum");
    m.def("warp_reduce_fp16_cuda", &warp_reduce_fp16_cuda, "warp reduce fp16");
}
