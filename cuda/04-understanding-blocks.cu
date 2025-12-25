#include <stdio.h>

/*
 * ============================================================================
 * KEY CONCEPTS
 * ============================================================================
 * - A block is a group of threads that can COOPERATE
 * - Threads in a block can synchronize with __syncthreads()
 * - Threads in a block share memory (shared memory - covered later)
 * - Max 1024 threads per block (hardware limitation)
 * - blockDim.x tells you how many threads are in your block
 * - Blocks are independent - different blocks cannot communicate directly
 */

// Kernel that demonstrates block size awareness
__global__ void showBlockInfo() {
    printf("Thread %d: I'm in a block with %d threads total\n",
           threadIdx.x, blockDim.x);
}

// Kernel that demonstrates why synchronization matters
// Two-stage computation: square elements, then sum with neighbor
__global__ void twoStageSum(int* array, int* result, int size) {
    int i = threadIdx.x;

    // Stage 1: Square each element
    array[i] = array[i] * array[i];

    // CRITICAL: Wait for ALL threads to finish stage 1 before starting stage 2
    // Stage 2 needs to read neighbor's squared value!
    __syncthreads();

    // Stage 2: Sum with next neighbor (skip last thread to avoid out of bounds)
    if (i < size - 1) {
        result[i] = array[i] + array[i + 1];
    }
}

// Kernel showing what happens WITHOUT synchronization (buggy on purpose)
__global__ void twoStageSumBuggy(int* array, int* result, int size) {
    int i = threadIdx.x;

    // Stage 1: Square each element
    array[i] = array[i] * array[i];

    // BUG: No __syncthreads() here!
    // Thread i might read array[i+1] BEFORE thread i+1 squares it!

    // Add artificial delay to odd threads to create race condition
    // This forces even threads to race ahead while odd threads are delayed
    if (i % 2 == 1) {
        volatile int dummy = 0;
        for (int j = 0; j < 100000; j++) {
            dummy += 1;
        }
    }

    // Stage 2: Sum with next neighbor
    // Even threads will reach here while odd threads are still delayed
    // So thread 0 reads array[1] BEFORE thread 1 has squared it!
    if (i < size - 1) {
        result[i] = array[i] + array[i + 1];
    }
}

int main() {
    printf("=== PART 1: What is a Block? ===\n\n");

    printf("A BLOCK is a group of threads that can COOPERATE:\n");
    printf("- Threads in same block can synchronize with __syncthreads()\n");
    printf("- Threads in same block can share memory (shared memory)\n");
    printf("- Threads in same block run on same hardware (Streaming Multiprocessor)\n");
    printf("- Blocks are INDEPENDENT - different blocks cannot communicate\n\n");

    printf("Hardware limitation: Max 1024 threads per block\n\n");

    printf("=== PART 2: Block Size Awareness ===\n\n");

    printf("Launching with <<<1, 8>>> (1 block, 8 threads):\n");
    showBlockInfo<<<1, 8>>>();
    cudaDeviceSynchronize();
    printf("\n");

    printf("Launching with <<<1, 16>>> (1 block, 16 threads):\n");
    showBlockInfo<<<1, 16>>>();
    cudaDeviceSynchronize();
    printf("\n");

    printf("Notice: blockDim.x tells each thread the block size\n\n");

    printf("=== PART 3: Why Synchronization Matters ===\n\n");

    // Two-stage computation demo
    int size = 8;
    int h_input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int h_result[8];

    int *d_array, *d_result;
    cudaMalloc(&d_array, size * sizeof(int));
    cudaMalloc(&d_result, size * sizeof(int));

    printf("Original array: [");
    for (int i = 0; i < size; i++) {
        printf("%d", h_input[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n\n");

    printf("Two-stage computation:\n");
    printf("  Stage 1: Square each element\n");
    printf("  Stage 2: Sum each element with its right neighbor\n\n");

    // Calculate expected results manually
    printf("Expected results:\n");
    printf("  After stage 1: [1, 4, 9, 16, 25, 36, 49, 64]\n");
    printf("  After stage 2: [1+4=5, 4+9=13, 9+16=25, 16+25=41, 25+36=61, 36+49=85, 49+64=113]\n\n");

    // Test 1: Correct version WITH synchronization
    cudaMemcpy(d_array, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    printf("Running WITH __syncthreads() (correct):\n");
    twoStageSum<<<1, size>>>(d_array, d_result, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, (size - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result: [");
    for (int i = 0; i < size - 1; i++) {
        printf("%d", h_result[i]);
        if (i < size - 2) printf(", ");
    }
    printf("]\n");
    printf("Correct! All values match expected results.\n\n");

    // Test 2: Buggy version WITHOUT synchronization
    cudaMemcpy(d_array, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    printf("Running WITHOUT __syncthreads() (buggy):\n");
    printf("(Odd threads artificially delayed to force race condition)\n");
    twoStageSumBuggy<<<1, size>>>(d_array, d_result, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, (size - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result: [");
    for (int i = 0; i < size - 1; i++) {
        printf("%d", h_result[i]);
        if (i < size - 2) printf(", ");
    }
    printf("]\n");
    printf("WRONG! Even-indexed results (0, 2, 4, 6) should be incorrect.\n");
    printf("They read neighbors' OLD values before squaring completed.\n");
    printf("Expected [5, 13, 25, 41, 61, 85, 113]\n\n");

    printf("=== PART 4: Block Limitations ===\n\n");

    printf("Maximum threads per block: 1024 (hardware limit)\n");
    printf("This means:\n");
    printf("- <<<1, 1024>>> is OK\n");
    printf("- <<<1, 2000>>> would FAIL (exceeds limit)\n\n");

    printf("If you need more than 1024 threads, you MUST use multiple blocks.\n");
    printf("That's what we'll cover in the next example!\n\n");

    cudaFree(d_array);
    cudaFree(d_result);

    printf("=== Summary ===\n");
    printf("Block = Group of cooperating threads\n");
    printf("- Can synchronize with __syncthreads()\n");
    printf("- Can share memory (shared memory)\n");
    printf("- Max 1024 threads per block\n");
    printf("- Independent from other blocks\n");

    return 0;
}

/*
 * ============================================================================
 * FREQUENTLY ASKED QUESTIONS
 * ============================================================================
 *
 * 1. What exactly is __syncthreads() doing?
 * ------------------------------------------
 * __syncthreads() is a barrier that makes all threads in a block WAIT until
 * every thread reaches that point.
 *
 * Think of it like a checkpoint in a race:
 * - Fast threads reach it first and wait
 * - Slow threads catch up
 * - Only when ALL threads arrive does anyone continue
 *
 * Why this matters for twoStageSum:
 *
 *   Stage 1:
 *   Thread 0: array[0] = 1*1 = 1
 *   Thread 1: array[1] = 2*2 = 4
 *   Thread 2: array[2] = 3*3 = 9
 *   ...
 *   __syncthreads();  ← Everyone waits here until stage 1 is done
 *
 *   Stage 2:
 *   Thread 0: result[0] = array[0] + array[1] = 1 + 4 = 5
 *   Thread 1: result[1] = array[1] + array[2] = 4 + 9 = 13
 *   ...
 *
 * Without __syncthreads():
 *   Thread 0: array[0] = 1*1 = 1
 *   Thread 0: result[0] = array[0] + array[1] = 1 + 2 = 3  ← WRONG!
 *             Thread 0 read array[1] BEFORE Thread 1 squared it!
 *             Got old value (2) instead of squared value (4)
 *
 * This is called a "race condition" - results depend on random timing.
 *
 *
 * 2. Why is there a 1024 thread limit per block?
 * -----------------------------------------------
 * Hardware limitation. Each block runs on a Streaming Multiprocessor (SM),
 * which has limited resources:
 * - Limited registers (variables)
 * - Limited shared memory
 * - Limited scheduling capacity
 *
 * 1024 threads per block is a balance between:
 * - Enough threads to keep GPU busy
 * - Not so many that resources are exhausted
 *
 * Different GPU architectures might have different limits, but 1024 is the
 * common modern limit.
 *
 *
 * 3. Can threads in different blocks communicate?
 * ------------------------------------------------
 * Not directly during kernel execution.
 *
 * Blocks are INDEPENDENT by design because:
 * - GPUs have many SMs (Streaming Multiprocessors)
 * - Blocks are distributed across SMs
 * - Blocks might run at different times (not simultaneously)
 * - No guarantee about block execution order
 *
 * If you need communication between blocks:
 * - Option 1: Use global memory (slow, no synchronization)
 * - Option 2: Split into multiple kernel launches (GPU finishes, CPU launches again)
 * - Option 3: Use atomic operations (advanced, covered later)
 *
 * This independence is actually a FEATURE - it allows GPUs to scale:
 * - GPU with 10 SMs can run 10 blocks simultaneously
 * - GPU with 100 SMs can run 100 blocks simultaneously
 * - Same code works on both without changes!
 *
 *
 * 4. When would I actually use __syncthreads()?
 * -----------------------------------------------
 * Common use cases:
 *
 * 1. Multi-stage algorithms where step 2 needs step 1's results:
 *    // Step 1: All threads compute something
 *    shared_data[threadIdx.x] = compute();
 *    __syncthreads();  // Wait for everyone to finish
 *    // Step 2: All threads can now safely read shared_data
 *    result = process(shared_data);
 *
 * 2. Parallel reduction (sum, max, min):
 *    Threads cooperatively combine values in multiple steps
 *    Each step needs __syncthreads() before the next
 *
 * 3. Shared memory access patterns:
 *    Loading data into shared memory, then processing it
 *
 * 4. In-place array modifications (like our reverse example):
 *    Separate read phase from write phase to avoid race conditions
 *
 * Key rule: If thread A's write might affect thread B's read, and both are
 * in the same block, you probably need __syncthreads() between them.
 *
 *
 * 5. What happens if I use __syncthreads() incorrectly?
 * -------------------------------------------------------
 * Common mistake: Using __syncthreads() inside an if statement
 *
 * BAD:
 *   if (threadIdx.x < 5) {
 *       __syncthreads();  // ❌ DEADLOCK!
 *   }
 *
 * Why this fails:
 * - Threads 0-4 reach __syncthreads() and wait
 * - Threads 5+ never reach __syncthreads()
 * - Threads 0-4 wait FOREVER (deadlock)
 *
 * GOOD:
 *   if (threadIdx.x < 5) {
 *       // do work
 *   }
 *   __syncthreads();  // ✓ All threads reach this
 *
 * Rule: ALL threads in the block must reach the same __syncthreads() call.
 * Never put it inside divergent code (if/else where some threads skip it).
 */
