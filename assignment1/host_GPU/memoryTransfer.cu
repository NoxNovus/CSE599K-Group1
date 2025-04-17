#include<cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

void initVector(float* vector, int size) {
    srand(1);
    for (int i = 0; i < size; i++) {
        vector[i] = (float)rand() / RAND_MAX;
    }
}

float measureTime(float* host_vector, int size) {
    float* device_vector;
    cudaMalloc((void**)&device_vector, size * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // start timing
    cudaEventRecord(start);

    // memory transfers
    // cudaMemcpy(device_vector, host_vector, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(host_vector, device_vector, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaFree(device_vector);
    // Print results
    return time;
}

int main(){
    // warmup
    for (int i = 0; i < 10; i++) {
        int warmupSize = 1024;
        float* host_warmupVector = (float *) malloc(warmupSize * sizeof(float));
        initVector(host_warmupVector, warmupSize);
        float* device_warmupVector;
        cudaMalloc((void**)&device_warmupVector, warmupSize * sizeof(float));

        cudaMemcpy(device_warmupVector, host_warmupVector, warmupSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(host_warmupVector, device_warmupVector, warmupSize * sizeof(float), cudaMemcpyDeviceToHost);

        free(host_warmupVector);
        cudaFree(device_warmupVector);
    }

    // iterations
    bool pinnedMemory = false;
    int num_bytes = 1;
    int num_iter = 25;
    float times[num_iter];
    for (int i = 0; i < num_iter; i ++) {
        float *vector;
        if (pinnedMemory) {
            cudaMallocHost(&vector, num_bytes * sizeof(float));
        } else {
            vector = (float *) malloc(num_bytes * sizeof(float));
        }
        initVector(vector, num_bytes);
        times[i] = measureTime(vector, num_bytes);
        num_bytes *= 2;
        if (pinnedMemory) {
            cudaFreeHost(vector);
        } else {
            free(vector);
        }
    }
    // graph bandwidth vs transfer size curve
    num_bytes = 1;
    for (int i = 0; i < num_iter; i ++) {
        printf("Transfer Size: %d \n", num_bytes);
        printf("Bandwidth: %f \n", num_bytes / times[i]);

        num_bytes *= 2;
    }
}