#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include "silu.h"

int main() {
    // Launch the kernel
    int dim = 8192;
    int num = dim * dim;

    // Allocate memory on the host
    // row major order (even though asisgment says we do SILU for a matrix, just treat it as a vector)
    float * h_input = new float[num];
    float * h_output = new float[num];

    // initialize host arrays [copied random # gen from online lol]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < num; i++) {
        h_input[i] = dis(gen);
    }

    // Allocate memory on the device
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, num * sizeof(float));
    cudaMalloc((void**)&d_output, num * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, num * sizeof(float), cudaMemcpyHostToDevice);
  
    // Warmup runs
    for (int i = 0; i < 10000; i++) {
        silu(d_input, d_output, num);
    }

    // benchmark (got this from online)
    int iter = 10000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start timing
    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) {
        silu(d_input, d_output, num);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    // Print results
    float iter_time = time / iter / 1000.0f;
    std::cout << "Avg Time: " << iter_time << " seconds \n";
    std::cout << "Bandwidth: " << 2 * num * sizeof(float) / iter_time / 1e9 << " GB/s\n";


   // error handling
   cudaDeviceSynchronize();
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return -1;
   }

   // Copy reuslts back to host
   cudaMemcpy(h_output, d_output, num * sizeof(float), cudaMemcpyDeviceToHost);

    //  Verify that results are correct
    for (int i = 0; i < num; i++) {  
        if ((h_input[i] / (1 + expf(-h_input[i]))) - h_output[i] > 10e-4) {
            std::cerr << "Error at index " << i << ": " << h_input[i] << std::endl;
            break;
        }
    }

    // benchmark the cuda implementaiton


   // Free device memory
   cudaFree(d_input);
   cudaFree(d_output);
   // Free host memory
   delete[] h_input;
   delete[] h_output;

   return 0;
}