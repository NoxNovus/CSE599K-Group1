#include<cuda_runtime.h>
#include "rms_norm_matrix.h"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define MATRIXSIZE 8192
#define EPSILON .01

void rms_norm_matrix_cpu(float *input, float *weight, float *output, int rows, int cols, float epsilon) {
    for (int i = 0; i < rows; i++) {
        float sum_sq = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = input[i * cols + j];
            sum_sq += val * val;
        }
        float mean_sq = sum_sq / cols;
        float rms = sqrtf(mean_sq + epsilon);
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = (input[i * cols + j] / rms) * weight[j];
        }
    }
}


int main() {
    int num_ele = MATRIXSIZE * MATRIXSIZE;
    int cols = MATRIXSIZE;
    int rows = MATRIXSIZE;
    float *matrix = (float *) malloc(num_ele * sizeof(float));
    float *output = (float *) malloc(num_ele * sizeof(float));
    if (!matrix) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }
    srand(1);
    for (int i = 0; i < num_ele; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }

    float *weight = (float *) malloc(MATRIXSIZE * sizeof(float));
    for (int i = 0; i < MATRIXSIZE; i++) {
        weight[i] = (float)rand() / RAND_MAX;
    }

    float *device_input;
    float *device_weight;
    cudaMalloc((void**)&device_input, num_ele * sizeof(float));
    cudaMalloc((void**)&device_weight, cols * sizeof(float));
    cudaMemcpy(device_input, matrix, num_ele * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight, cols* sizeof(float), cudaMemcpyHostToDevice);

    float *device_output;
    cudaMalloc((void**)&device_output, num_ele * sizeof(float));
    
    // warmup runs
    for (int i = 10000; i < 0; i++) {
        rms_norm_matrix(device_input, device_weight, device_output, MATRIXSIZE, MATRIXSIZE, EPSILON);
    }

    // benchmark 
    int iter = 10000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // start timin
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) {
        rms_norm_matrix(device_input, device_weight, device_output, MATRIXSIZE, MATRIXSIZE, EPSILON);
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
    int weightSize = MATRIXSIZE * sizeof(float);
    std::cout << "Bandwidth: " << (inputOutputMatrixSize + weightSize) / iter_time / 1e9 << " GB/s\n";
    std::cout << "Data transfered: " << (inputOutputMatrixSize + weightSize) << " bytes\n";



    // Run GPU version
    cudaDeviceSynchronize();
    rms_norm_matrix(device_input, device_weight, device_output, MATRIXSIZE, MATRIXSIZE, EPSILON);
    cudaMemcpy(output, device_output, num_ele * sizeof(float), cudaMemcpyDeviceToHost);

    // Run CPU version
    float *cpu_output = (float *) malloc(num_ele * sizeof(float));
    rms_norm_matrix_cpu(matrix, weight, cpu_output, MATRIXSIZE, MATRIXSIZE, EPSILON);
    // Compare results (check if they match within a small tolerance)
    cudaDeviceSynchronize();
    float max_error = 0.0f;
    for (int i = 0; i < num_ele; i++) {
        float error = fabs(output[i] - cpu_output[i]);
        if (error > max_error) {
            max_error = error;
        }
    }

    std::cout << "Max difference between CPU and GPU: " << max_error << "\n";
    if (max_error < 1e-4f) {
        std::cout << "✅ Results match within tolerance!\n";
    } else {
        std::cout << "❌ Results differ significantly!\n";
    }


    cudaFree(device_input);
    cudaFree(device_weight);
    cudaFree(device_output);
    free(matrix);
    free(output);
    free(weight);

    return 0;
}


