import numpy as np
import time
import sys
sys.path.append("./build")

from binding import embedding_cuda, embedding_cuda_fp16_pack
from python.embedding import embedding_py

def benchmark(func, *args):
    time.sleep(1)
    start = time.time()
    func(*args)
    end = time.time()
    return (end - start) * 1000

N = 1 << 20
VocabSize = 10000
D = 128


indices = np.random.randint(0, VocabSize, size=N, dtype=np.int32)
weight = np.random.randn(VocabSize, D).astype(np.float32)
output = np.zeros((N, D), dtype=np.float32)

weight_fp16 = np.random.randn(VocabSize, D).astype(np.float16)
output_fp16 = np.zeros((N, D), dtype=np.float16)

print(f"CUDA time: {embedding_cuda(indices, weight, output):.8f} ms")
print(f"CUDA fp16 time: {embedding_cuda(indices, weight_fp16, output_fp16):.8f} ms")
print(f"Python time: {benchmark(embedding_py, indices, weight):.8f} ms")