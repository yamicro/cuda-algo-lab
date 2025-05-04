import numpy as np
import sys
import time
sys.path.append("./build")

from binding import sigmoid_cuda, sigmoid_cuda_fp16_pack
from python.sigmod import sigmoid_py

def benchmark(func, *args):
    time.sleep(1)
    start = time.time()
    func(*args)
    end = time.time()
    return (end - start) * 1000

N = 1 << 20
x = np.random.rand(N).astype(np.float32)
y = np.zeros(N, dtype=np.float32)
x_fp16 = np.random.randn(N).astype(np.float16)
y_fp16 = np.zeros_like(x).astype(np.float16)

print(f"CUDA time: {sigmoid_cuda(x, y):.8f} ms")
print(f"CUDA fp16 pack time: {sigmoid_cuda_fp16_pack(x_fp16, y_fp16, N):.8f} ms")
print(f"Python time: {benchmark(sigmoid_py,  x):.8f} ms")