import numpy as np

def histogram_py(a, y):
    for i in a:
        y[i] += 1
    return y