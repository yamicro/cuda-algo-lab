import sys
sys.path.append("./build") 

import numpy as np
import time
from binding import add_cuda, add_fp16_pack_cuda
from python.add import add_py
import os

sys.path.append(os.path.abspath("build"))
sys.path.append(os.path.abspath("."))

def benchmark(func, arc_type, *args):
    for _ in range(2):
        func(*args)
    time.sleep(1)
    start = time.time()
    func(*args)
    end = time.time()
    return (end - start) * 1000


N = 1 << 20
x = np.random.rand(N).astype(np.float32)
y = np.random.rand(N).astype(np.float32)
z = np.zeros(N, dtype=np.float32)
x_fp16 = np.random.rand(N).astype(np.float16)
y_fp16 = np.random.rand(N).astype(np.float16)


print(f"CUDA time: {add_cuda(x, y, z):.8f} ms")
print(f"CUDA fp16 pack time: {add_fp16_pack_cuda(x_fp16, y_fp16):.8f} ms")
print(f"Python time: {benchmark(add_py, "python", x, y):.8f} ms")
