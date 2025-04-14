#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include "cuda/add.cu"
#include "cuda/histgram.cu"

void add_cuda(pybind11::array_t<float> a, pybind11::array_t<float> b, pybind11::array_t<float> c) {
    auto buf_a = a.unchecked<1>();
    auto buf_b = b.unchecked<1>();
    auto buf_c = c.mutable_unchecked<1>();
    int n = buf_a.size();

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));

    cudaMemcpy(d_a, buf_a.data(0), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, buf_b.data(0), n * sizeof(float), cudaMemcpyHostToDevice);

    add_kernel<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);

    cudaMemcpy(buf_c.mutable_data(0), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}


void histogram_cuda(pybind11::array_t<int> a, pybind11::array_t<int> y) {
    auto buf_a = a.unchecked<1>();
    auto buf_y = y.mutable_unchecked<1>();
    int N = buf_a.size();

    int *d_a, *d_y;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_y, y.size() * sizeof(int));

    cudaMemcpy(d_a, buf_a.data(0), N * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_y, buf_y.data(0), buf_y.size(0) * sizeof(int), cudaMemcpyHostToDevice);

    histogram_kernel<<<(N+255)/256, 256>>>(d_a, d_y, N);

    cudaMemcpy(buf_y.mutable_data(0), d_y, buf_y.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_y);
}

PYBIND11_MODULE(binding, m) {
    m.def("add_cuda", &add_cuda, "CUDA add two arrays");
    m.def("histogram_cuda", &histogram_cuda, "CUDA histogram");
}
