#include <stdio.h>

// __global__ means this function runs on the GPU (device)
// and can be called from CPU (host)
__global__ void helloFromGPU() {
    // This code executes on the GPU
    printf("Hello from GPU thread!\n");
}

int main() {
    // This code runs on the CPU (host)
    printf("Hello from CPU!\n");

    // Launch 1 GPU thread to run helloFromGPU()
    // Syntax: functionName<<<number_of_blocks, threads_per_block>>>()
    helloFromGPU<<<1, 1>>>();

    // Wait for GPU to finish before accessing results
    // Without this, the program might end before GPU prints
    cudaDeviceSynchronize();

    printf("Back to CPU!\n");

    return 0;
}

/*
 * HOW TO COMPILE AND RUN:
 *
 * nvcc 01-hello-gpu.cu -o 01-hello-gpu
 * ./01-hello-gpu
 *
 * EXPECTED OUTPUT:
 * Hello from CPU!
 * Hello from GPU thread!
 * Back to CPU!
 *
 * KEY CONCEPTS:
 * - __global__ = function that runs on GPU
 * - <<<1, 1>>> = launch config (1 block, 1 thread)
 * - cudaDeviceSynchronize() = wait for GPU to finish
 */
