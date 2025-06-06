#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(int *a, int *b, int *c, size_t num) {
    int block_start = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    int index = block_start + thread_id;
    if (index < num) {
        c[index] = a[index] + b[index];
    }
}


torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    // Check if the input tensors are on the same device
    if (a.device() != b.device()) {
        throw std::runtime_error("Input tensors must be on the same device");
    }

    // Check if the input tensors are 1D
    if (a.dim() != 1 || b.dim() != 1) {
        throw std::runtime_error("Input tensors must be 1D");
    }

    // Check if the input tensors have the same size
    if (a.size(0) != b.size(0)) {
        throw std::runtime_error("Input tensors must have the same size");
    }

    // Check if the input tensors are of type int
    if (a.dtype() != torch::kInt || b.dtype() != torch::kInt) {
        throw std::runtime_error("Input tensors must be of type int");
    }
    // Check if the input tensors are on the GPU
    if (a.device().is_cpu() || b.device().is_cpu()) {
        throw std::runtime_error("Input tensors must be on the GPU");
    }

    auto num = a.size(0);
    auto c = torch::empty_like(a);

    int threads_per_block = 256;
    int blocks_per_grid = (num + threads_per_block - 1) / threads_per_block;

    add_kernel<<<blocks_per_grid, threads_per_block>>>(a.data_ptr<int>(), b.data_ptr<int>(), c.data_ptr<int>(), num);
    cudaDeviceSynchronize();
    return c;
}

PYBIND11_MODULE(my_addition, m) {
    m.def("add", &add, "Add two tensors");
}