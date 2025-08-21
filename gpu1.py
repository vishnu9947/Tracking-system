import torch
import time
if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matrix_size = 5000
a = torch.randn(matrix_size, matrix_size, device=device)
b = torch.randn(matrix_size, matrix_size, device=device)
start_time = time.time()
result = torch.matmul(a, b)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Matrix multiplication time: {elapsed_time:.4f} seconds")
del a, b, result
torch.cuda.empty_cache()
