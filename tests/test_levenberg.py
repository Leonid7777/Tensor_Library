import numpy as np
import scipy.linalg as lg
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from  src.tensorlibrary import levenberg, cp_to_tensor

sizes = np.array((10, 20, 30))
T = np.zeros(sizes)
for i in range(sizes[0]):
    for j in range(sizes[1]):
        for k in range(sizes[2]):
                    T[i, j, k] = 1 / (i + j + k + 1)

rank = 2
start_time = time.time()
factors, it = levenberg(T, rank)
print(f'time = {time.time() - start_time}s')

new_T = cp_to_tensor(factors)
print(np.linalg.norm((new_T - T).reshape(-1,)) / np.linalg.norm(T.reshape(-1,)))