#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCKSIZE 1024

__global__ void scale_kernel(float *input, float *output, float *weight, int cols, float *RMS) {
    float scal_RMS = sqrt(*RMS / cols + .01);

    int block_start = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    int index = block_start + thread_id;
    output[index] = input[index] / scal_RMS * weight[index];
}

__global__ void square_elements_reduction_add_fused(float *input, float *output) {
    extern __shared__ float shared_data[];

    int block_start = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    int index = block_start + thread_id;
    shared_data[thread_id] = input[index] * input[index];
    // perform first level of reduction,
    // reading from global memory, writing to shared memory 
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_id < s) {
            shared_data[thread_id] += shared_data[thread_id + s];
        }
        __syncthreads();
    }
    // shared memory has final results, write back to global memory
    if (thread_id == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

__global__ void reduction_add(float *input) {
    extern __shared__ float shared_data[];

    // each thread loads one element from global to shared memory
    int block_start = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    int index = block_start + thread_id;
    shared_data[thread_id] = input[index];
    // perform first level of reduction,

    // reading from global memory, writing to shared memory 
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_id < s) {
            shared_data[thread_id] += shared_data[thread_id + s];
        }
        __syncthreads();
    }
    // shared memory has final results, write back to global memory
    if (thread_id == 0) {
        input[blockIdx.x] = shared_data[0];
    }
}

// assume matrix is row-major
// cols = 1024 * 1024 = 1048576
void rms_norm_vector(float *input, float *weight, float *output, int cols, float epsilon) {
    // summation
    static float* device_summation = nullptr;
    if (device_summation == nullptr){
        cudaMalloc((void**)&device_summation, cols / BLOCKSIZE * sizeof(float));
    }
    int num = cols;

    dim3 square_elements_block((num + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 square_elements_threads(BLOCKSIZE);

    square_elements_reduction_add_fused<<<square_elements_block, square_elements_threads, BLOCKSIZE * sizeof(float)>>>(input, device_summation);

    // summation in GPU
    int num_add = num / BLOCKSIZE;
    for (int i = 0; num_add >= BLOCKSIZE; i++) {
        num_add = num_add / BLOCKSIZE;
        dim3 num_block(num_add);
        dim3 num_threads(BLOCKSIZE);
        reduction_add<<<num_block, num_threads, BLOCKSIZE * sizeof(float)>>>(device_summation);
    }
    // we now perform the scale operation
    int scale_BLOCKSIZE = 256;
    dim3 scale_block((num + scale_BLOCKSIZE - 1) / scale_BLOCKSIZE);
    dim3 scale_threads(scale_BLOCKSIZE);
    scale_kernel<<<scale_block, scale_threads>>>(input, output, weight, cols, device_summation);
}