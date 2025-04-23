import numpy as np
import time
import sys
sys.path.append("./build")

from binding import gelu_cuda
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

print(f"CUDA time: {gelu_cuda(x, y):.4f} ms")
print(f"Python time: {benchmark(gelu_py, x):.4f} ms")