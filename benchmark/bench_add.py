import sys
sys.path.append("./build") 

import numpy as np
import time
from binding import add_cuda  # 来自 pybind11 binding
from python.add import add_py  # 纯 Python baseline


def benchmark(func, *args):
    start = time.time()
    func(*args)
    end = time.time()
    return (end - start) * 1000  # ms


N = 1 << 20  # 1048576 个 float
x = np.random.rand(N).astype(np.float32)
y = np.random.rand(N).astype(np.float32)
z = np.zeros(N, dtype=np.float32)

print(f"CUDA time: {benchmark(add_cuda, x, y, z):.4f} ms")
print(f"Python time: {benchmark(add_py, x, y):.4f} ms")
