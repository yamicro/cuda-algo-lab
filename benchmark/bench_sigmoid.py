import numpy as np
import sys
import time
sys.path.append("./build")

from binding import sigmoid_cuda
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

print(f"CUDA time: {sigmoid_cuda(x, y):.4f} ms")
print(f"Python time: {benchmark(sigmoid_py,  x):.4f} ms")