import numpy as np

def mat_transpose_py(input):
    return [[input[j][i] for j in range(len(input))] for i in range(len(input[0]))]