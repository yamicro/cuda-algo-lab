import sys
sys.path.append("./build")

import numpy as np
import time
from binding import histogram_cuda
from python.histogram import histogram_py
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
    return (end - start) * 1000  # ms


N = 1 << 20
max_value = 100
x = np.random.randint(0, max_value, size=N, dtype=np.int32)
y = np.zeros(max_value, dtype=np.int32)

print(f"CUDA time: {histogram_cuda(x, y):.8f} ms")
print(f"Python time: {benchmark(histogram_py, x, y):.8f} ms")