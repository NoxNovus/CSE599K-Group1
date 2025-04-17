#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "copy_first_column.h"

void initA(float *d_A, int num) {
    srand(1);
    for (int i = 0; i < num; i++) {
        d_A[i] = (float)rand() / RAND_MAX;
    }
}

int main(){
    // initialize array
    int rows = 8192;
    int cols = 65536;
    float* h_A = (float *) malloc(rows * cols * sizeof(float));
    float* temp_h_A = (float *) malloc(rows * cols * sizeof(float));
    initA(h_A, rows * cols);
    float *d_A;
    cudaMalloc(&d_A, rows * cols * sizeof(float));

    // warmup
    for (int i = 0; i < 10000; i++) {
        copy_first_column(h_A, d_A, rows, cols);
    }
    
    // benchmark (got this from online)
    int iter = 10000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // start timing
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) {
        copy_first_column(h_A, d_A, rows, cols);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    float iter_time = time / iter;
    std::cout << "Avg Time: " << iter_time << " milliseconds \n";

    // verify copy worked
    // move device back to temp in host
    cudaDeviceSynchronize();
    cudaMemcpy(temp_h_A, d_A, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < rows; i++) {
        if (h_A[cols * i] != temp_h_A[cols * i]) {
            printf("%f %f \n", h_A[cols * i] , temp_h_A[cols * i]);
            // this fails!
            break;
        }
    }
}