import sys
sys.path.append("./build") 

import numpy as np
import time
from binding import add_cuda
from python.add import add_py


def benchmark(func, arc_type, *args):
    if arc_type is "python":
        time.sleep(1)
        start = time.time()
        func(*args)
        end = time.time()
        return (end - start) * 1000
    else:
        import torch
        torch.cuda.synchronize()
        start = time.time()
        func(*args)
        torch.cuda.synchronize()
        end = time.time()
        return (end - start) * 1000


N = 1 << 20
x = np.random.rand(N).astype(np.float32)
y = np.random.rand(N).astype(np.float32)
z = np.zeros(N, dtype=np.float32)

print(f"CUDA time: {add_cuda(x, y, z):.4f} ms")
print(f"Python time: {benchmark(add_py, "python", x, y):.4f} ms")
