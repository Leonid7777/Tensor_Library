import numpy as np
import scipy.linalg as lg
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from  src.tensorlibrary import als

sizes = np.array((10, 20, 30))
T = np.zeros(sizes)
for i in range(sizes[0]):
    for j in range(sizes[1]):
        for k in range(sizes[2]):
                    T[i, j, k] = 1 / (i + j + k + 1)

rank = 1
start_time = time.time()
factors, it = als(T, rank)
print(f'time = {time.time() - start_time}s')

len_dim = len(sizes)
sweep_l = np.ones(rank).reshape(1, -1)

for j in range(len_dim):
    sweep_l = lg.khatri_rao(sweep_l, factors[j])

print(np.linalg.norm(np.sum(sweep_l, axis=1) - T.reshape(-1,)) / np.linalg.norm(T.reshape(-1,)))