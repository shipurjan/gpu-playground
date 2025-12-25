#include <stdio.h>
#include <stdlib.h>  // for malloc, free
#include <string.h>  // for memcpy

/*
 * ============================================================================
 * KEY CONCEPTS
 * ============================================================================
 * - malloc/free work on CPU memory (system RAM)
 * - cudaMalloc/cudaFree work on GPU memory (VRAM)
 * - memcpy copies within CPU memory
 * - cudaMemcpy copies BETWEEN CPU and GPU memory
 * - CPU cannot directly access GPU memory (separate address spaces)
 */

// Simple kernel that doubles each element in an array
__global__ void doubleArray(int* array, int size) {
    // threadIdx.x = unique thread ID within a block (0, 1, 2, 3, 4...)
    // Each thread gets a different ID so they can work on different array elements
    int i = threadIdx.x;
    printf("  [GPU Thread %d] Processing array[%d] = %d -> %d\n",
           i, i, array[i], array[i] * 2);

    // Safety check: Prevent out-of-bounds access if more threads than elements
    // Example: If someone launches <<<1, 10>>> but size=5, threads 5-9 skip work
    // In this example we launch exactly 5 threads for 5 elements, so this seems
    // redundant, but it's standard practice to always include this bounds check
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
    printf("   Values: [%d, %d, %d, %d, %d]\n\n",
           cpu_array[0], cpu_array[1], cpu_array[2], cpu_array[3], cpu_array[4]);

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
    //
    // What free() actually does:
    // - Marks that memory as "available for reuse" by the system
    // - Does NOT zero out the memory
    // - Does NOT set the pointer to NULL
    // - The pointer still holds the same address (now a "dangling pointer")
    //
    // After free(), accessing that memory is UNDEFINED BEHAVIOR:
    // - Might still show old values (if memory hasn't been reused)
    // - Might show garbage (if something else used that memory)
    // - Might crash your program
    // - Results are unpredictable!

    printf("3. free() - what it actually does:\n");
    printf("   Before free():\n");
    printf("     Address: %p\n", (void*)cpu_copy);
    printf("     Values:  [%d, %d, %d, %d, %d]\n",
           cpu_copy[0], cpu_copy[1], cpu_copy[2], cpu_copy[3], cpu_copy[4]);

    free(cpu_copy);

    printf("   After free():\n");
    printf("     Address: %p (SAME! Pointer not set to NULL)\n", (void*)cpu_copy);
    printf("     Values:  [%d, %d, %d, %d, %d] (UNDEFINED BEHAVIOR!)\n",
           cpu_copy[0], cpu_copy[1], cpu_copy[2], cpu_copy[3], cpu_copy[4]);
    printf("     ^ These might be old values, garbage, or cause a crash\n");
    printf("     ^ Accessing freed memory is a BUG - don't do this!\n\n");

    printf("   What free() did:\n");
    printf("     - Told the OS: \"This memory can be reused\"\n");
    printf("     - Did NOT zero out the memory\n");
    printf("     - Did NOT set cpu_copy to NULL\n");
    printf("     - Memory might get overwritten by other allocations\n\n");

    printf("   Best practice after free():\n");
    cpu_copy = NULL;  // Manually set to NULL to avoid dangling pointer
    printf("     cpu_copy = NULL; // Now it's %p (safe!)\n", (void*)cpu_copy);
    printf("     Accessing NULL will crash immediately (better than undefined behavior)\n\n");

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
    printf("   Sent: [%d, %d, %d, %d, %d]\n\n",
           cpu_array[0], cpu_array[1], cpu_array[2], cpu_array[3], cpu_array[4]);

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
 * ============================================================================
 * FREQUENTLY ASKED QUESTIONS
 * ============================================================================
 *
 * 1. How does threadIdx work? Is the kernel called once or multiple times?
 * -------------------------------------------------------------------------
 * When you launch a kernel with <<<1, 5>>>, the function runs 5 times IN PARALLEL.
 *
 * It's NOT called 5 times sequentially like a for-loop. Instead:
 * - CUDA launches 5 threads simultaneously
 * - Each thread executes the SAME function code
 * - Each thread gets a DIFFERENT threadIdx.x value (0, 1, 2, 3, 4)
 * - threadIdx.x is automatically assigned by the GPU
 * - threadIdx.x is local to each thread (each sees their own value)
 *
 * Example:
 *   doubleArray<<<1, 5>>>(gpu_array, 5);
 *
 * What happens:
 *   Thread 0: runs doubleArray(), sees threadIdx.x = 0, processes array[0]
 *   Thread 1: runs doubleArray(), sees threadIdx.x = 1, processes array[1]
 *   Thread 2: runs doubleArray(), sees threadIdx.x = 2, processes array[2]
 *   Thread 3: runs doubleArray(), sees threadIdx.x = 3, processes array[3]
 *   Thread 4: runs doubleArray(), sees threadIdx.x = 4, processes array[4]
 *   ^ All of these run AT THE SAME TIME (in parallel)
 *
 * This is why GPUs are fast: Instead of processing 5 elements one-by-one,
 * they process all 5 simultaneously using different threads.
 *
 * Think of it like this:
 *   CPU for-loop: One worker processes 5 items sequentially
 *   GPU kernel:   5 workers process 5 items simultaneously
 *
 *
 * 2. What are the limitations of GPU threads? Can they do any work CPU threads can?
 * ---------------------------------------------------------------------------------
 * GPU threads CAN execute the same code as CPU threads (loops, conditionals, math),
 * but there are important performance differences and some limitations.
 *
 * You can write almost ANY code, but some patterns are MUCH slower on GPU:
 *
 * **What GPU threads CAN do:**
 * - Loops: for/while loops work fine
 * - Branching: if/else statements work
 * - Complex math: floating point, trigonometry, etc.
 * - Function calls: calling other __device__ functions
 * - Limited recursion (modern CUDA, but not recommended)
 *
 * **Performance characteristics:**
 *
 * 1. BRANCHING is expensive (warp divergence):
 *    GPU threads execute in groups of 32 called "warps"
 *    All threads in a warp run the SAME instruction simultaneously
 *
 *    if (threadIdx.x % 2 == 0) {
 *        // expensive computation A
 *    } else {
 *        // expensive computation B
 *    }
 *
 *    Result: GPU must run BOTH branches. Threads not taking a branch sit idle.
 *    You lose 50% efficiency here. Branching is allowed but hurts performance.
 *
 * 2. LOOPS with different iteration counts hurt performance:
 *    If thread 0 loops 10 times and thread 1 loops 1000 times,
 *    thread 0 sits idle waiting for thread 1 to finish.
 *
 * 3. SEQUENTIAL DEPENDENCIES are bad:
 *    GPUs excel at parallel work, struggle with sequential chains.
 *    Bad: result[i] = result[i-1] + 1  (each step depends on previous)
 *    Good: result[i] = input[i] * 2     (all independent)
 *
 * **"Simple work" is misleading terminology:**
 * - It doesn't mean "easy math" - GPUs can do complex calculations
 * - It means "PARALLEL work" - same operation on many data elements
 * - Example of complex GPU work: physics simulation on 100,000 particles
 * - Each particle's calculation can be complex, but they're independent
 *
 * **CPU vs GPU thread comparison:**
 *   CPU (8-16 threads):
 *   - Heavy threads (large stack, lots of cache)
 *   - Great at sequential logic, branching, unpredictable control flow
 *   - Example: traversing a tree, parsing JSON, complex business logic
 *
 *   GPU (thousands of threads):
 *   - Lightweight threads (small stack, less cache per thread)
 *   - Great at doing the SAME operation on massive datasets
 *   - Example: image processing, matrix math, array operations
 *
 * **When to use GPU:**
 * ✓ Same operation on massive amounts of data
 * ✓ Minimal branching or predictable branches
 * ✓ Independent computations (no dependencies between elements)
 * ✓ Enough work to justify CPU→GPU copy overhead
 *
 * **When to use CPU:**
 * ✓ Irregular/unpredictable branching
 * ✓ Sequential dependencies (step N needs result from step N-1)
 * ✓ Tree/graph traversal, linked lists
 * ✓ Small datasets (GPU copy overhead not worth it)
 *
 *
 * 3. Is there threadIdx.y and threadIdx.z? What other built-in variables exist?
 * ------------------------------------------------------------------------------
 * Yes! threadIdx is a 3D structure with .x, .y, and .z components.
 * CUDA provides several built-in variables for thread organization:
 *
 * **Thread identification:**
 * - threadIdx.x, threadIdx.y, threadIdx.z
 *   Your thread's position within its block (0 to blockDim-1)
 *
 * **Block identification:**
 * - blockIdx.x, blockIdx.y, blockIdx.z
 *   Your block's position within the grid (0 to gridDim-1)
 *
 * **Dimensions:**
 * - blockDim.x, blockDim.y, blockDim.z
 *   Number of threads per block in each dimension
 *
 * - gridDim.x, gridDim.y, gridDim.z
 *   Number of blocks in the grid in each dimension
 *
 * **Why use 2D/3D organization?**
 * Match your thread layout to your data structure:
 *
 * 1D (arrays):
 *   int data[1000];
 *   kernel<<<10, 100>>>();  // 10 blocks, 100 threads each
 *   int i = blockIdx.x * blockDim.x + threadIdx.x;  // Global index
 *   data[i] = ...;
 *
 * 2D (images/matrices):
 *   int image[HEIGHT][WIDTH];
 *   dim3 blocks(16, 16);    // 16x16 threads per block
 *   dim3 grid(WIDTH/16, HEIGHT/16);
 *   kernel<<<grid, blocks>>>();
 *
 *   In kernel:
 *   int x = blockIdx.x * blockDim.x + threadIdx.x;  // Column
 *   int y = blockIdx.y * blockDim.y + threadIdx.y;  // Row
 *   image[y][x] = ...;
 *
 * 3D (volumes, 3D simulations):
 *   int volume[DEPTH][HEIGHT][WIDTH];
 *   dim3 blocks(8, 8, 8);   // 8x8x8 = 512 threads per block
 *   dim3 grid(WIDTH/8, HEIGHT/8, DEPTH/8);
 *   kernel<<<grid, blocks>>>();
 *
 *   In kernel:
 *   int x = blockIdx.x * blockDim.x + threadIdx.x;
 *   int y = blockIdx.y * blockDim.y + threadIdx.y;
 *   int z = blockIdx.z * blockDim.z + threadIdx.z;
 *   volume[z][y][x] = ...;
 *
 * **Our example uses only threadIdx.x because:**
 * - We have a 1D array (5 elements)
 * - We launch with <<<1, 5>>> = 1 block, 5 threads in x-dimension
 * - threadIdx.y and threadIdx.z would both be 0 (unused dimensions)
 *
 * You'll see 2D/3D thread organization in image processing and matrix examples.
 *
 *
 * 4. How can I tell if a pointer's address is on CPU or GPU?
 * ------------------------------------------------------------
 * You can't easily tell by looking at the address value alone. Here's why:
 *
 * CPU and GPU have SEPARATE address spaces:
 * - A GPU pointer (like 0x7f8b4c000000) doesn't map to CPU memory
 * - A CPU pointer (like 0x7ffc0726b6c4) doesn't map to GPU memory
 * - They're completely independent memory systems
 *
 * Common approaches to track pointer locations:
 *
 * 1. Naming convention (recommended):
 *    int* h_array;  // h_ prefix = host (CPU)
 *    int* d_array;  // d_ prefix = device (GPU)
 *
 * 2. What happens if you get it wrong:
 *    int* gpu_ptr;
 *    cudaMalloc(&gpu_ptr, size);
 *    int value = *gpu_ptr;  // ❌ CRASH! Can't dereference GPU pointer from CPU
 *
 *    int* cpu_ptr = malloc(size);
 *    cudaMemcpy(result, cpu_ptr, size, cudaMemcpyDeviceToHost);  // ❌ Wrong direction
 *
 * 3. Programmatic check (advanced):
 *    cudaPointerAttributes attrs;
 *    cudaPointerGetAttributes(&attrs, ptr);
 *    // Check attrs.type to see if it's host or device memory
 *
 * Best practice: Use h_ and d_ prefixes consistently to avoid confusion.
 */
