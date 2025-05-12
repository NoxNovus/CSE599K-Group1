#include "copy_first_column.h"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

void copy_first_column(float *h_A, float *d_A, int rows, int cols) {
    // convert to cpu list first
    // summation
    static float* host_first_column = nullptr;
    if (host_first_column == nullptr) {
        cudaMallocHost((void**)&host_first_column, sizeof(float) * rows);
    }

    for (int i = 0; i < rows; i ++) {
        host_first_column[i] = h_A[i * cols];
    }
    // do the memory copy
    cudaMemcpy(d_A, host_first_column, sizeof(float) * rows, cudaMemcpyHostToDevice);
}