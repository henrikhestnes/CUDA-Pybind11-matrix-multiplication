import numpy as np
import time
import matplotlib.pyplot as plt

import sys
sys.path.append('build')
import gpu_library


M, N, K = 32*8, 32*8, 32*8
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
print("Python time: " + str(end-start))


print(f"\nCPU in C++ result correct: {np.allclose(C_CPU.reshape(M, K),C_PYTHON)}")
print(f"GPU with only global memory correct: {np.allclose(C_GPU_GLOBAL.reshape(M, K),C_PYTHON)}")
print(f"GPU with shared memory correct: {np.allclose(C_GPU_SHARED.reshape(M, K),C_PYTHON)}")