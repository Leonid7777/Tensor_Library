import numpy as np
import scipy.linalg as lg
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from  src.tensorlibrary import TT_SVD, TT_to_tensor

sizes = np.array((10, 20, 30, 40))
T = np.zeros(sizes)
for i in range(sizes[0]):
    for j in range(sizes[1]):
        for k in range(sizes[2]):
            for l in range(sizes[3]):
                T[i, j, k, l] = 1 / (i + j + k + l + 1)

factors, ranks = TT_SVD(T.copy(), eps=10**(-10))

new_T, dims = TT_to_tensor(factors.copy(), ranks.copy())

print(np.linalg.norm((new_T - T).reshape(-1,)) / np.linalg.norm((T).reshape(-1,)))