import numpy as np
import sys
import time
sys.path.append("./build")

from binding import warp_reduce_sum_cuda
from python.warp_reduce import warp_reduce_py

def benchmark(func, *args):
    time.sleep(1)
    start = time.time()
    func(*args)
    end = time.time()
    return (end - start) * 1000

N = 1 << 15  # 必须是 32 的倍数
x = np.random.randn(N).astype(np.float32)
y = np.zeros(N // 32, dtype=np.float32)

print(f"CUDA time: {warp_reduce_sum_cuda(x, y):.4f} ms")
print(f"Python time: {benchmark(warp_reduce_py, x):.4f} ms")
