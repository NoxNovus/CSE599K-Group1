#include<cuda_runtime.h>
#include "rms_norm_vector.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#define EPSILON .01

void rms_norm_vector_cpu(float *input, float *weight, float *output, int cols, float epsilon) {
    // Compute mean squared value for the current row
    float sum_sq = 0.0f;
    for (int i = 0; i < cols; i++) {
        float val = input[i];
        sum_sq += val * val;
    }
    float mean_sq = sum_sq / cols;
    float rms = sqrtf(mean_sq + epsilon);

    // Normalize the row and scale by weight
    for (int i = 0; i < cols; i++) {
        output[i] = (input[i] / rms) * weight[i];
    }
}


int main() {
    // row major
    int num_ele = 1024 * 1024;
    int cols = num_ele;
    float *vector = (float *) malloc(num_ele * sizeof(float));
    float *output = (float *) malloc(num_ele * sizeof(float));
    if (!vector) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    // Seed the random # generator
    srand(1);
    for (int i = 0; i < num_ele; i++) {
        vector[i] = (float)rand() / RAND_MAX;
        // vector[i] = 1;
    }

    float *weight = (float *) malloc(cols * sizeof(float));
    for (int i = 0; i < cols; i++) {
        weight[i] = (float)rand() / RAND_MAX;
    }

    float *device_input;
    float *device_weight;
    cudaMalloc((void**)&device_input, num_ele * sizeof(float));
    cudaMalloc((void**)&device_weight, cols * sizeof(float));
    cudaMemcpy(device_input, vector, num_ele * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight, cols* sizeof(float), cudaMemcpyHostToDevice);

    float *device_output;
    cudaMalloc((void**)&device_output, num_ele * sizeof(float));
    
    // warmup runs
    for (int i = 1; i < 10000; i++) {
        rms_norm_vector(device_input, device_weight, device_output, num_ele, EPSILON);
    }

    int iter = 10000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // start timin
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) {
        rms_norm_vector(device_input, device_weight, device_output, num_ele, EPSILON);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    // Print results
    cudaMemcpy(output, device_output, num_ele * sizeof(float), cudaMemcpyDeviceToHost);
    float iter_time = time / iter / 1000.0f;
    std::cout << "Avg Time: " << iter_time << " seconds \n";
    int inputOutputMatrixSize = 2 * num_ele * sizeof(float);
    int weightSize = cols * sizeof(float);
    std::cout << "Bandwidth: " << (inputOutputMatrixSize + weightSize) / iter_time / 1e9 << " GB/s\n";
    std::cout << "Data transfered: " << (inputOutputMatrixSize + weightSize) << " bytes\n";



    // // Run GPU version
    // rms_norm_vector(device_input, device_weight, device_output, num_ele, EPSILON);
    // cudaMemcpy(output, device_output, num_ele * sizeof(float), cudaMemcpyDeviceToHost);

    // // Run CPU version
    // float *cpu_output = (float *) malloc(num_ele * sizeof(float));
    // rms_norm_vector_cpu(vector, weight, cpu_output, cols, EPSILON);
    // // Compare results (check if they match within a small tolerance)
    // float max_error = 0.0f;
    // for (int i = 0; i < num_ele; i++) {
    //     float error = fabs(output[i] - cpu_output[i]);
    //     if (error > max_error) {
    //         max_error = error;
    //     }
    // }

    // std::cout << "Max difference between CPU and GPU: " << max_error << "\n";
    // if (max_error < 1e-2f) {
    //     std::cout << "✅ Results match within tolerance!\n";
    // } else {
    //     std::cout << "❌ Results differ significantly!\n";
    // }


    cudaFree(device_input);
    cudaFree(device_weight);
    cudaFree(device_output);
    free(vector);
    free(output);
    free(weight);

    return 0;
}


