import torch
import torch.nn.functional as F
from silu_triton_kernel import silu_triton

# Test the Triton kernel
torch.manual_seed(1)
device = torch.device('cuda')
matrix_dim = 8192
num_elements = matrix_dim ** 2
matrix = torch.rand(num_elements, device = 'cpu').reshape((matrix_dim, matrix_dim))

matrix = matrix.to('cuda')
output_torch = F.silu(matrix).cpu()
output_triton = silu_triton(matrix).cpu()
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')


# warmup 
def warmup(func, input):
    for i in range(10000):
        func(input)

# Benchmark the Triton kernel
num_iter = 10000
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# perform SILU on the tensors
for i in range(num_iter):
    result = silu_triton(matrix)
end.record()
torch.cuda.synchronize()

each_iter_time = start.elapsed_time(end) / 1000 / num_iter
print("Time taken for silu triton:", each_iter_time, "seconds")
print("Bandwidth: ", 2 * matrix.element_size() * matrix.numel() / each_iter_time / 1e9, "GB/s")