import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from  src.tensorlibrary import rsvd

n = 2000
m = 1000
matrix = np.zeros((n, m))

for i in range(n):
    for j in range(m):
        matrix[i, j] = 1 / (i + j + 1)

start_time = time.time()
U, s, V = rsvd(matrix, 24)
end_time = time.time()
print(f"time = {end_time - start_time}")

print(np.linalg.norm((U * s) @ V - matrix) / np.linalg.norm(matrix))