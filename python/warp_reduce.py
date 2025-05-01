import numpy as np

def warp_reduce_py(x):
    x = x.reshape(-1, 32)
    return x.sum(axis=1)