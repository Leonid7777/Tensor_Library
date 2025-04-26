import numpy as np
import scipy.linalg as lg
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from  src.tensorlibrary import TT_SVD, TT_to_tensor, TT_orthogonalization

sizes = np.array((10, 20, 30, 40))
T = np.zeros(sizes)
for i in range(sizes[0]):
    for j in range(sizes[1]):
        for k in range(sizes[2]):
            for l in range(sizes[3]):
                T[i, j, k, l] = 1 / (i + j + k + l + 1)

factors, ranks = TT_SVD(T.copy(), eps=10**(-10))

for i in range(len(factors)):
    factors[i] += np.ones(factors[i].shape)

print("factors are not orthogonal")
print(np.linalg.norm((factors[0].T @ factors[0] - np.eye(factors[0].shape[1])).reshape(-1,)) / np.linalg.norm((np.eye(factors[0].shape[1])).reshape(-1,)))
for i in range(1, len(factors) - 1):
    a, b, c = factors[i].shape
    new_fact = factors[i].reshape(a * b, c)
    print(np.linalg.norm((new_fact.T @ new_fact - np.eye(c)).reshape(-1,)) / np.linalg.norm((np.eye(c)).reshape(-1,)))

new_T, dims = TT_to_tensor(factors.copy(), ranks.copy())
ort_factors, ort_ranks = TT_orthogonalization(factors, ranks)

print("now factors are orthogonal")

print(np.linalg.norm((ort_factors[0].T @ ort_factors[0] - np.eye(ort_factors[0].shape[1])).reshape(-1,)) / np.linalg.norm((np.eye(ort_factors[0].shape[1])).reshape(-1,)))
for i in range(1, len(ort_factors) - 1):
    a, b, c = ort_factors[i].shape
    new_fact = ort_factors[i].reshape(a * b, c)
    print(np.linalg.norm((new_fact.T @ new_fact - np.eye(c)).reshape(-1,)) / np.linalg.norm((np.eye(c)).reshape(-1,)))

ort_T, dims = TT_to_tensor(ort_factors.copy(), ort_ranks.copy())

print(f'Error = {np.linalg.norm((new_T - ort_T).reshape(-1,)) / np.linalg.norm((new_T).reshape(-1,))}')