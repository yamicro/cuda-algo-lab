import numpy as np
import sys
import time
sys.path.append("./build")

from binding import mat_transpose_cuda, mat_x4_transpose_cuda
from python.mat_transpose import mat_transpose_py

def benchmark(func, *args):
    for _ in range(2):
        func(*args)
    time.sleep(1)
    start = time.time()
    func(*args)
    end = time.time()
    return (end - start) * 1000

M = 1024
N = 2048
#
x = np.random.randn(M, N).astype(np.float32)
y = np.zeros((N, M), dtype=np.float32)
#
print(f"CUDA time: {mat_transpose_cuda(x, y):.8f} ms")
print(f"CUDA for 4 pack time: {mat_x4_transpose_cuda(x, y):.8f} ms")
print(f"Python time: {benchmark(mat_transpose_py,  x):.8f} ms")