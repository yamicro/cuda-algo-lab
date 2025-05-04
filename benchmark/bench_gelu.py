import numpy as np
import time
import sys

sys.path.append("./build")

from binding import gelu_cuda, gelu_cuda_fp16_pack
from python.gelu import gelu_py

def benchmark(func, *args):
    time.sleep(1)
    start = time.time()
    func(*args)
    end = time.time()
    return (end - start) * 1000

N = 1 << 20
x = np.random.randn(N).astype(np.float32)
y = np.zeros_like(x)

x_fp16 = np.random.randn(N).astype(np.float16)
y_fp16 = np.zeros_like(x).astype(np.float16)

print(f"CUDA time: {gelu_cuda(x, y):.8f} ms")
print(f"CUDA fp16 pack time: {gelu_cuda_fp16_pack(x_fp16, y_fp16):.8f} ms")
print(f"Python time: {benchmark(gelu_py, x):.8f} ms")