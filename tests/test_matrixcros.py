import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from  src.tensorlibrary import cross_approx

n = 1000
m = 2000
matrix = np.zeros((n, m))

for i in range(n):
    for j in range(m):
        matrix[i, j] = 1 / (i + j + 1)

start_time = time.time()
Q, R, rank = cross_approx(matrix, 30, eps = 10**(-10))
end_time = time.time()
print(f"time = {end_time - start_time}")

print(np.linalg.norm(Q @ R - matrix) / np.linalg.norm(matrix))
print(f'rank = {rank}')