import numpy as np

def sigmoid_py(x):
    return 1 / (1 + np.exp(-x))