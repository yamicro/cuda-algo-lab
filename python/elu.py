import numpy as np

def elu_py(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))