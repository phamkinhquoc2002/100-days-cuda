import torch
import time

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vector size
N = 1024

# Allocate and initialize vectors on GPU
A = torch.ones(N, device=device)
B = torch.ones(N, device=device)

# --- Timing GPU execution only ---
torch.cuda.synchronize()  # Ensure everything before timing is done
start = time.time()

C = A + B

torch.cuda.synchronize()  # Wait for all GPU ops to finish
end = time.time()

# Display time in milliseconds
print(f"PyTorch GPU Time: {(end - start) * 1000:.4f} ms")

# Print first 5 elements for validation
print("C[:5] =", C[:5].tolist())
