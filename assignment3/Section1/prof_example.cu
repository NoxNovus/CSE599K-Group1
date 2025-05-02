
#include <cuda_runtime.h>
#include <iostream>

#define BLOCKSIZE 256


// Kernel function to add two vectors
__global__ void add(int *a, int *b, int *c, size_t num) {
    int block_start = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    int index = block_start + thread_id;
    if (index < num) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    size_t num = 100000;

    int * host_a = new int[num];
    int * host_b = new int[num];
    int * host_c = new int[num];

    // Initialize host arrays
    for (int i = 0; i < num; i++) {
        host_a[i] = i;
        host_b[i] = i;
    }
    

    // Allocate memory on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, num * sizeof(int));
    cudaMalloc((void**)&d_b, num * sizeof(int));
    cudaMalloc((void**)&d_c, num * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_a, host_a, num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, host_b, num * sizeof(int), cudaMemcpyHostToDevice);

    dim3 num_block((num + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 num_threads(BLOCKSIZE);

    
    int warm_up_count = 100;
    int profile_count = 100;
    size_t L2_size = 50 * 1024 * 1024;

    for (int i = 0; i < warm_up_count; ++i)
    {
        add<<<num_block, num_threads>>>(d_a, d_b, d_c, num);
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    int* clear_l2_buffer;
    cudaMalloc(&clear_l2_buffer, L2_size);

    float total_ms = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < profile_count; ++i)
    {
        cudaMemset(clear_l2_buffer, 0, L2_size); // Clear L2 cache https://github.com/NVIDIA/nvbench/blob/main/nvbench/detail/l2flush.cuh
        cudaEventRecord(start);
        add<<<num_block, num_threads>>>(d_a, d_b, d_c, num);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    
    float average_time = total_ms / profile_count; 
    std::cout << "Average time: " << average_time << " ms" << std::endl;

    // Free the L2 buffer
    cudaFree(clear_l2_buffer);
    // Free CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] host_a;
    delete[] host_b;
    delete[] host_c;

    return 0;
}

