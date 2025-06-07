import numpy as np
import sys
import time
from binding import elu_cuda, elu_cuda_fp16_pack
from python.elu import elu_py
import os

sys.path.append(os.path.abspath("build"))
sys.path.append(os.path.abspath("."))
def benchmark(func, *args):
    for _ in range(2):
        func(*args)
    time.sleep(1)
    start = time.time()
    func(*args)
    end = time.time()
    return (end - start) * 1000

N = 1 << 20
x = np.random.randn(N).astype(np.float32)
y = np.zeros_like(x)
alpha = 1.0
x_fp16 = np.random.randn(N).astype(np.float16)


print(f"CUDA time: {elu_cuda(x, y) :.8f} ms")
print(f"CUDA fp16 pack time: {elu_cuda_fp16_pack(x_fp16):.8f} ms")
print(f"Python time: {benchmark(elu_py, x, alpha):.8f} ms")