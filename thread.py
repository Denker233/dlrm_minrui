import torch
import os

print("OMP_NUM_THREADS:", os.getenv("OMP_NUM_THREADS"))  # Check if it's set
print("PyTorch Threads:", torch.get_num_interop_threads())  # Get current thread count

import torch
# torch.set_num_threads(40)
# print("PyTorch Threads:", torch.get_num_threads())
# print("PyTorch Threads:", torch.get_num_threads())
