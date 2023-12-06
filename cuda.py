import numpy as np
from numba import cuda

# Function for vector addition using CUDA
@cuda.jit
def add_vectors_gpu(a, b, result):
    i = cuda.grid(1)
    result[i] = a[i] + b[i]

# Length of the vector
vector_length = 1000

# Create input vectors
a = np.random.rand(vector_length)
b = np.random.rand(vector_length)

# Initialize the result vector
result = np.zeros_like(a)

# Configure CUDA threads
threads_per_block = 256
blocks_per_grid = (vector_length + (threads_per_block - 1)) // threads_per_block

# Execute the CUDA kernel
add_vectors_gpu[blocks_per_grid, threads_per_block](a, b, result)

# Print the results
print("Input Vector A:", a)
print("Input Vector B:", b)
print("Result Vector:", result)
