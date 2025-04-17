#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCKSIZE 256
#define MATRIXSIZE 8192
__global__ void scale_kernel(float *input, float *output, float *weight, int cols, int rows, float *RMS) {
    int block_start = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    int index = block_start + thread_id;
    int col_idx = index % cols;
    int row_idx = index / cols;
    output[index] = input[index] / RMS[row_idx] * weight[col_idx];
}

__global__ void reduction_add_sm(float *input, float *weight, float *output) {
    extern __shared__ float shared_data[];

    // each thread loads one element from global to shared memory
    int block_start = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    int index = block_start + thread_id;
    shared_data[thread_id] = input[index] * input[index];
    // perform first level of reduction,

    // reading from global memory, writing to shared memory 
    __syncthreads();
    if (thread_id == 0) {
        float partial_sum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            partial_sum += shared_data[i];
        }
        output[blockIdx.x] = partial_sum;
    }
}

// numCols should be equal to 
__global__ void reduction_add_gm(float* semi_coalesed_sum, float* vector_RMS, int numToCoalescePerRow, int cols, float epsilon) {
    // each thread is responsible for coalescing a single row into its summation
    int block_start = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    int row_num = block_start + thread_id;
    
    // now global memory has the final results, coalesce into a single sum
    float row_sum = 0;
    for (int i = 0; i < numToCoalescePerRow; i++) {
        row_sum += semi_coalesed_sum[i + row_num * numToCoalescePerRow];
    }
    vector_RMS[row_num] = sqrt(row_sum / cols + epsilon);
}

// assume matrix is row-major
// We have MATRIXSIZE = cols = rows = 8192
void rms_norm_matrix(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    int num = rows * cols;
    int int_num_block = (num + BLOCKSIZE - 1) / BLOCKSIZE;


    dim3 num_block(int_num_block);
    dim3 half_num_block(int_num_block);
    dim3 num_threads(BLOCKSIZE);
    // gpu kernel:
    // this will load the matrix to shared memory and then square each value
    // this will then do a reduction add on shared memory, with many threads responsible for coalescing each partition of a row
    // writes the reduction to global memory via output (which also temporarly serves as device_semi_coalesed_sum)
    static float* device_vector_RMS = nullptr;
    if (device_vector_RMS == nullptr){
        cudaMalloc((void**)&device_vector_RMS, rows * sizeof(float));
    }

    reduction_add_sm<<<num_block, num_threads, BLOCKSIZE * sizeof(float)>>>(input, weight, output);

    // gpu kernel:
    // we then do a reduction add on global memory, with one thread reponsible for coalescing each row
    dim3 sm_num_blocks(MATRIXSIZE / BLOCKSIZE);
    dim3 sm_num_threads(BLOCKSIZE);

    int numToCoalescePerRow = cols / BLOCKSIZE;
    reduction_add_gm<<<sm_num_blocks, sm_num_threads>>>(output, device_vector_RMS, numToCoalescePerRow, cols, epsilon);

    scale_kernel<<<num_block, num_threads>>>(input, output, weight, cols, rows, device_vector_RMS);
}