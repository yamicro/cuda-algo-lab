import numpy as np
import time
import sys
sys.path.append("./build")

from binding import sgemv_k32_cuda, sgemv_k128_cuda
from python.sgemv import sgemv_py

def benchmark(func, *args):
    time.sleep(1)
    start = time.time()
    func(*args)
    end = time.time()
    return (end - start) * 1000

M = 1024
K = 128  # 对 sgemv_k128 要保证是 128 的倍数

a = np.random.rand(M, K).astype(np.float32)
x = np.random.rand(K).astype(np.float32)
y1 = np.zeros(M, dtype=np.float32)
y2 = np.zeros(M, dtype=np.float32)

print(f"K32 CUDA:  {sgemv_k32_cuda(a, x, y1):.8f} ms")
print(f"K128 CUDA: {sgemv_k128_cuda(a, x, y2):.8f} ms")
print(f"Python:    {benchmark(sgemv_py, a, x):.8f} ms")
