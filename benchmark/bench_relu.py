import numpy as np
import sys
import time
sys.path.append("./build")

from binding import relu_cuda
from python.relu import relu_py

def benchmark(func, *args):
    time.sleep(1)
    start = time.time()
    func(*args)
    end = time.time()
    return (end - start) * 1000

N = 1 << 20
x = np.random.rand(N).astype(np.float32)
y = np.zeros(N, np.float32)

print(f"CUDA time: {relu_cuda(x, y):.4f} ms")
print(f"Python time: {benchmark(relu_py,  x):.4f} ms")