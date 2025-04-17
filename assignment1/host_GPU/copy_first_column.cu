#include "copy_first_column.h"
#include <cuda_runtime.h>
void copy_first_column(float *h_A, float *d_A, int rows, int cols) {
    // byte‑pitch of each row on host = k * sizeof(float)
    size_t spitch = cols * sizeof(float);
    size_t width_bytes = 1 * sizeof(float);
    size_t height = rows;
    size_t dpitch = cols * sizeof(float);

    cudaMemcpy2D(
    d_A,                
    dpitch,            
    h_A,                
    spitch,                 
    width_bytes,            
    height,                 
    cudaMemcpyHostToDevice  
    );
}