import torch
from torch.profiler import profile, record_function, ProfilerActivity
def silu(x):
    return x / (1 + torch.exp(-x))

def warmup(func, input):
    for i in range(10000):
        func(input)

if __name__ == "__main__":
    device = torch.device('cuda')
    matrix_dim = 8192
    num_elements = matrix_dim ** 2
    matrix = torch.rand(num_elements, device = 'cpu').reshape((matrix_dim, matrix_dim))

    matrix = matrix.to('cuda')

    # warmup
    warmup(silu, matrix)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    # benchmark
    num_iter = 10000
    for i in range(num_iter):
        result = silu(matrix)
    end.record()
    torch.cuda.synchronize()

    each_iter_time = start.elapsed_time(end) / 1000 / num_iter # milliseconds -> seconds

    print("Time taken for silu:", each_iter_time, "seconds")

    transferToDevice = matrix.element_size() * matrix.numel()
    transferToHost = matrix.element_size() * matrix.numel()
    totalTransfer = transferToDevice + transferToHost
    print("Bandwidth: ", totalTransfer / each_iter_time / 1e9, "GB/s")

    result = result.cpu()



# import torch
# from torch.profiler import profile, record_function, ProfilerActivity
# def silu(x):
#     return x / (1 + torch.exp(-x))

# def warmup(func, input):
#     for i in range(10):
#         func(input)

# if __name__ == "__main__":
#     device = torch.device('cuda')
#     matrix_dim = 8192
#     num_elements = matrix_dim ** 2
#     matrix = torch.rand(num_elements, device = 'cpu').reshape((matrix_dim, matrix_dim))

#     matrix = matrix.to('cuda')

#     # warmup
#     warmup(silu, matrix)
#     with profile(
#         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         profile_memory=True,
#         with_stack=True,
#         record_shapes=True,
#     ) as prof:
#         # Use record_function to mark the section of the code to be profiled
#         with record_function("silu"):
#             # Add the tensors
#             for i in range(100):
#                 result = silu(matrix)

#     # export to chrome trace
#     prof.export_chrome_trace("trace.json")
#     result = result.cpu()
#     # Print the result
#     print("Result of addition:", result)