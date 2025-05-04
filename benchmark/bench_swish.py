import numpy as np
import sys
import time
sys.path.append("./build")

from binding import swish_cuda, swish_cuda_fp16_pack
from python.swish import swish_py

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
y_fp16 = np.zeros_like(x_fp16)  # 确保是 float16

print(f"CUDA time: {swish_cuda(x, y):.8f} ms")
# 下面多传一个参数 N
print(f"CUDA fp16 pack time: {swish_cuda_fp16_pack(x_fp16, y_fp16):.8f} ms")
print(f"Python time: {benchmark(swish_py,  x):.8f} ms")
