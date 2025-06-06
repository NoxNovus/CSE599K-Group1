import sys
sys.path.append('./build')
import torch
import my_addition

a = torch.ones(4096, device='cuda', dtype=torch.int32)
b = torch.ones(4096, device='cuda', dtype=torch.int32)

# Test the custom CUDA kernel
c = my_addition.add(a, b)
print(c)