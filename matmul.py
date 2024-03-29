import numpy as np
import time

import sys
sys.path.append('build')
import gpu_library

np.random.seed(0)
M, N, K = 32*5, 32*5, 32*5

A = np.random.rand(M, N)
B = np.random.rand(N, K)

start = time.perf_counter()
C_CPU = np.zeros(M*K)
gpu_library.cpu_matrix_multiply(A.reshape(M*N), B.reshape(N*K), C_CPU, M, N, K)
end = time.perf_counter()
print("CPU in C++ time: " + str(end-start))


start = time.perf_counter()
C_GPU_GLOBAL = np.zeros(M*K)
gpu_library.cuda_global_matrix_multiply(A.reshape(M*N), B.reshape(N*K), C_GPU_GLOBAL, M, N, K)
end = time.perf_counter()
print("GPU with only global memory time: " + str(end-start))


start = time.perf_counter()
C_GPU_SHARED = np.zeros(M*K)
gpu_library.cuda_shared_matrix_multiply(A.reshape(M*N), B.reshape(N*K), C_GPU_SHARED, M, N, K)
end = time.perf_counter()
print("GPU with shared memory time: " + str(end-start))


start = time.perf_counter()
C_PYTHON = A @ B
end = time.perf_counter()
print("Python optmimized time: " + str(end-start))


start = time.perf_counter()
C_PYTHON_NATIVE = [[0 for _ in range(K)] for _ in range(M)]
for i in range(M):
    for j in range(K):
        for k in range(N):
            C_PYTHON_NATIVE[i][j] += A[i][k] * B[k][j]
end = time.perf_counter()
print("Python native time: " + str(end-start))


print(f"\nCPU in C++ result correct: {np.allclose(C_CPU.reshape(M, K),C_PYTHON)}")
print(f"GPU with only global memory correct: {np.allclose(C_GPU_GLOBAL.reshape(M, K),C_PYTHON)}")
print(f"GPU with shared memory correct: {np.allclose(C_GPU_SHARED.reshape(M, K),C_PYTHON)}")
print(f"Python native correct: {np.allclose(np.array(C_PYTHON_NATIVE).reshape(M, K),C_PYTHON)}")