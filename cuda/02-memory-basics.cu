#include <stdio.h>
#include <stdlib.h>  // for malloc, free
#include <string.h>  // for memcpy

/*
 * This example demonstrates memory management in C and CUDA.
 * We'll create an array, copy it, and free it - first on CPU, then on GPU.
 */

// Simple kernel that doubles each element in an array
__global__ void doubleArray(int* array, int size) {
    int i = threadIdx.x;
    if (i < size) {
        array[i] = array[i] * 2;
    }
}

int main() {
    printf("=== PART 1: C Memory Management (CPU only) ===\n\n");

    // ============================================================
    // malloc() - Allocate memory on the CPU (heap)
    // ============================================================
    // Returns a pointer to allocated memory
    // You must free() it when done to avoid memory leaks

    int* cpu_array = (int*)malloc(5 * sizeof(int));
    printf("1. malloc() allocated memory on CPU\n");
    printf("   Address: %p\n", (void*)cpu_array);

    // Initialize the array
    cpu_array[0] = 10;
    cpu_array[1] = 20;
    cpu_array[2] = 30;
    cpu_array[3] = 40;
    cpu_array[4] = 50;
    printf("   Values: [10, 20, 30, 40, 50]\n\n");

    // ============================================================
    // memcpy() - Copy memory from one CPU location to another
    // ============================================================
    // Syntax: memcpy(destination, source, num_bytes)
    // This copies data within CPU memory

    int* cpu_copy = (int*)malloc(5 * sizeof(int));
    memcpy(cpu_copy, cpu_array, 5 * sizeof(int));
    printf("2. memcpy() copied array to another CPU location\n");
    printf("   Original: %p\n", (void*)cpu_array);
    printf("   Copy:     %p\n", (void*)cpu_copy);
    printf("   Copy values: [%d, %d, %d, %d, %d]\n\n",
           cpu_copy[0], cpu_copy[1], cpu_copy[2], cpu_copy[3], cpu_copy[4]);

    // ============================================================
    // free() - Deallocate memory on CPU
    // ============================================================
    // Always free what you malloc to avoid memory leaks

    free(cpu_copy);
    printf("3. free() deallocated the copy from CPU memory\n\n");

    printf("=== PART 2: CUDA Memory Management (CPU + GPU) ===\n\n");

    // ============================================================
    // cudaMalloc() - Allocate memory on the GPU
    // ============================================================
    // Syntax: cudaMalloc(&pointer, num_bytes)
    // Note: Takes a POINTER to a pointer (&gpu_array)
    // Returns a pointer that points to GPU memory (not accessible from CPU!)

    int* gpu_array;
    cudaMalloc(&gpu_array, 5 * sizeof(int));
    printf("4. cudaMalloc() allocated memory on GPU\n");
    printf("   GPU address: %p (not accessible from CPU!)\n", (void*)gpu_array);
    printf("   Size: %zu bytes\n\n", 5 * sizeof(int));

    // ============================================================
    // cudaMemcpy() - Copy memory between CPU and GPU
    // ============================================================
    // Syntax: cudaMemcpy(destination, source, num_bytes, direction)
    // Direction can be:
    //   - cudaMemcpyHostToDevice (CPU -> GPU)
    //   - cudaMemcpyDeviceToHost (GPU -> CPU)
    //   - cudaMemcpyDeviceToDevice (GPU -> GPU)

    // Copy FROM CPU TO GPU
    cudaMemcpy(gpu_array, cpu_array, 5 * sizeof(int), cudaMemcpyHostToDevice);
    printf("5. cudaMemcpy() copied data from CPU to GPU\n");
    printf("   Direction: Host (CPU) -> Device (GPU)\n");
    printf("   Sent: [10, 20, 30, 40, 50]\n\n");

    // Run kernel to double the values on GPU
    printf("6. Launching kernel to double values on GPU...\n");
    doubleArray<<<1, 5>>>(gpu_array, 5);
    cudaDeviceSynchronize();
    printf("   Kernel finished\n\n");

    // Copy FROM GPU TO CPU
    int* result = (int*)malloc(5 * sizeof(int));
    cudaMemcpy(result, gpu_array, 5 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("7. cudaMemcpy() copied results from GPU back to CPU\n");
    printf("   Direction: Device (GPU) -> Host (CPU)\n");
    printf("   Received: [%d, %d, %d, %d, %d]\n\n",
           result[0], result[1], result[2], result[3], result[4]);

    // ============================================================
    // cudaFree() - Deallocate memory on GPU
    // ============================================================
    // Just like free(), but for GPU memory

    cudaFree(gpu_array);
    printf("8. cudaFree() deallocated GPU memory\n\n");

    // Clean up remaining CPU memory
    free(cpu_array);
    free(result);

    printf("=== Summary ===\n");
    printf("CPU functions (work on CPU memory):\n");
    printf("  - malloc()  : allocate\n");
    printf("  - memcpy()  : copy within CPU\n");
    printf("  - free()    : deallocate\n\n");
    printf("GPU functions (work on GPU memory):\n");
    printf("  - cudaMalloc()  : allocate on GPU\n");
    printf("  - cudaMemcpy()  : copy between CPU <-> GPU\n");
    printf("  - cudaFree()    : deallocate from GPU\n\n");
    printf("Key difference: CPU and GPU have SEPARATE memory!\n");
    printf("You cannot directly access GPU memory from CPU code.\n");

    return 0;
}

/*
 * HOW TO COMPILE AND RUN:
 *
 * nvcc 02-memory-basics.cu -o 02-memory-basics
 * ./02-memory-basics
 *
 * EXPECTED OUTPUT:
 * Shows step-by-step memory allocation, copying, and deallocation
 * on both CPU and GPU, with memory addresses and values.
 *
 * KEY CONCEPTS:
 * - malloc/free work on CPU memory (system RAM)
 * - cudaMalloc/cudaFree work on GPU memory (VRAM)
 * - memcpy copies within CPU memory
 * - cudaMemcpy copies BETWEEN CPU and GPU memory
 * - CPU cannot directly access GPU memory (separate address spaces)
 */
