import numpy as np

def swish_py(x):
    return x/(1+np.exp(-x))