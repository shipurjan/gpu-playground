#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * ============================================================================
 * KEY CONCEPTS
 * ============================================================================
 * - Matrix multiplication is O(N³) - perfect for GPU parallelization
 * - Each output element is independent (no dependencies)
 * - Small matrices: CPU wins (GPU overhead not worth it)
 * - Large matrices: GPU dominates (parallelism beats overhead)
 * - Uses 2D thread grid: threadIdx.x/y + blockIdx.x/y
 * - Demonstrates when GPU compute actually pays off
 */

// CPU matrix multiplication (naive implementation)
void matmulCPU(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// GPU matrix multiplication kernel
// Each thread computes one element of the output matrix
__global__ void matmulGPU(float* A, float* B, float* C, int N) {
    // Calculate global row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: make sure we're within matrix bounds
    if (row < N && col < N) {
        float sum = 0.0f;

        // Compute dot product of row from A and column from B
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// Helper function to initialize matrix with random values
void initMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = (float)(rand() % 10);  // Random values 0-9
    }
}

// Helper function to verify results match (approximately)
bool verifyResults(float* cpu, float* gpu, int N) {
    for (int i = 0; i < N * N; i++) {
        if (abs(cpu[i] - gpu[i]) > 0.01f) {
            return false;
        }
    }
    return true;
}

// Benchmark function for a given matrix size
void benchmark(int N) {
    printf("\n=== Matrix Size: %dx%d (%d elements, %.2f MB) ===\n",
           N, N, N * N, (N * N * sizeof(float)) / (1024.0f * 1024.0f));

    // Allocate host memory
    size_t bytes = N * N * sizeof(float);
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C_cpu = (float*)malloc(bytes);
    float* h_C_gpu = (float*)malloc(bytes);

    // Initialize matrices
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // CPU computation
    printf("\nCPU computation...\n");
    clock_t cpu_start = clock();
    matmulCPU(h_A, h_B, h_C_cpu, N);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU time: %.2f ms\n", cpu_time);

    // GPU computation
    printf("\nGPU computation...\n");

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy input matrices to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    // Use 16x16 threads per block (common choice)
    int BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Grid: %dx%d blocks, Block: %dx%d threads, Total threads: %d\n",
           numBlocks.x, numBlocks.y,
           threadsPerBlock.x, threadsPerBlock.y,
           numBlocks.x * numBlocks.y * threadsPerBlock.x * threadsPerBlock.y);

    // GPU timing (using CPU clock for simplicity)
    clock_t gpu_start = clock();

    matmulGPU<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    clock_t gpu_end = clock();
    double gpu_time = ((double)(gpu_end - gpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("GPU time: %.2f ms\n", gpu_time);

    // Copy result back to host
    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    if (verifyResults(h_C_cpu, h_C_gpu, N)) {
        printf("Results verified: CPU and GPU match!\n");
    } else {
        printf("ERROR: Results don't match!\n");
    }

    // Performance comparison
    double speedup = cpu_time / gpu_time;
    printf("\nPerformance:\n");
    printf("  Speedup: %.2fx %s\n", speedup,
           speedup > 1.0 ? "(GPU faster)" : "(CPU faster)");

    if (speedup < 1.0) {
        printf("  CPU won because overhead of copying data to GPU\n");
        printf("  isn't worth it for this small problem size.\n");
    } else {
        printf("  GPU won because parallelism overcame memory copy overhead.\n");
        printf("  %d threads computed %d elements simultaneously!\n",
               numBlocks.x * numBlocks.y * threadsPerBlock.x * threadsPerBlock.y,
               N * N);
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    srand(time(NULL));

    printf("=== Matrix Multiplication: CPU vs GPU ===\n");
    printf("Demonstrates when GPU compute pays off\n");

    // Small matrix - CPU should win
    benchmark(32);

    // Large matrix - GPU should dominate
    benchmark(1024);

    printf("\n=== Summary ===\n");
    printf("Matrix multiplication is O(N³) - perfect for GPU parallelism\n");
    printf("1024x1024 matrix = 1,048,576 threads computing simultaneously!\n");

    return 0;
}

/*
 * ============================================================================
 * FREQUENTLY ASKED QUESTIONS
 * ============================================================================
 *
 * 1. Why is matrix multiplication a good GPU problem?
 * ----------------------------------------------------
 * Three key reasons:
 *
 * 1. Massive parallelism:
 *    - N×N output matrix = N² elements to compute
 *    - Each element is INDEPENDENT (no dependencies)
 *    - Can compute all N² elements simultaneously
 *    - 1024×1024 = 1,048,576 parallel computations!
 *
 * 2. Compute-heavy:
 *    - Each output element: N multiplications + N additions
 *    - Total operations: O(N³)
 *    - 1024×1024: ~1 billion operations
 *    - High computation-to-memory ratio (good for GPU)
 *
 * 3. Regular memory access:
 *    - Predictable access patterns
 *    - GPU can optimize memory fetches
 *    - (Advanced: can use shared memory for even better performance)
 *
 * This is why matrix multiplication is everywhere in ML, graphics, science!
 *
 *
 * 2. Why does CPU win for small matrices?
 * ----------------------------------------
 * GPU has overhead costs:
 *
 * 1. Memory copy time:
 *    - Copy A and B from CPU to GPU (cudaMemcpy)
 *    - Copy C from GPU back to CPU
 *    - For 64×64 floats: ~32 KB copied each way
 *    - This takes time!
 *
 * 2. Kernel launch overhead:
 *    - GPU needs to schedule threads
 *    - Set up execution context
 *    - Small fixed cost (~microseconds)
 *
 * 3. Not enough work to saturate GPU:
 *    - 64×64 = 4,096 elements
 *    - Modern GPU has thousands of cores
 *    - Each core sits idle most of the time
 *
 * For small problems, overhead > parallelism benefit, so CPU wins.
 *
 * Break-even point varies by hardware but typically:
 * - < 100×100: CPU competitive
 * - 100×100 to 500×500: Mixed (depends on hardware)
 * - > 500×500: GPU starts to dominate
 * - > 1000×1000: GPU massively faster
 *
 *
 * 3. What is the 2D thread indexing math?
 * ----------------------------------------
 * We organize threads in a 2D grid to match matrix structure:
 *
 * Thread organization:
 *   - Block: 16×16 threads (256 threads per block)
 *   - Grid: Enough blocks to cover entire matrix
 *
 * Each thread calculates its position:
 *   row = blockIdx.y * blockDim.y + threadIdx.y
 *   col = blockIdx.x * blockDim.x + threadIdx.x
 *
 * Example with 48×48 matrix, 16×16 blocks:
 *
 * Grid: 3×3 blocks (need 3 blocks of 16 to cover 48)
 *
 *   Block (0,0)    Block (1,0)    Block (2,0)
 *   ┌────────┐    ┌────────┐    ┌────────┐
 *   │ 0-15   │    │ 16-31  │    │ 32-47  │   rows 0-15
 *   └────────┘    └────────┘    └────────┘
 *
 *   Block (0,1)    Block (1,1)    Block (2,1)
 *   ┌────────┐    ┌────────┐    ┌────────┐
 *   │ 0-15   │    │ 16-31  │    │ 32-47  │   rows 16-31
 *   └────────┘    └────────┘    └────────┘
 *
 *   Block (0,2)    Block (1,2)    Block (2,2)
 *   ┌────────┐    ┌────────┐    ┌────────┐
 *   │ 0-15   │    │ 16-31  │    │ 32-47  │   rows 32-47
 *   └────────┘    └────────┘    └────────┘
 *
 * Thread in block (1,2), position (3,7):
 *   row = 2 * 16 + 7 = 39
 *   col = 1 * 16 + 3 = 19
 *   Computes C[39][19]
 *
 * Boundary check (if row < N && col < N) handles cases where matrix size
 * isn't evenly divisible by block size.
 *
 *
 * 4. Why use 16×16 threads per block? Why not 32×32?
 * ---------------------------------------------------
 * Several hardware constraints:
 *
 * 1. Max threads per block: 1024
 *    - 16×16 = 256 threads ✓
 *    - 32×32 = 1024 threads ✓ (at the limit)
 *    - 64×64 = 4096 threads ✗ (exceeds limit!)
 *
 * 2. Warp size: 32 threads
 *    - GPU executes threads in groups of 32 (warps)
 *    - 16×16 = 256 = 8 warps (efficient)
 *    - 32×32 = 1024 = 32 warps (also fine)
 *
 * 3. Register and shared memory pressure:
 *    - Each thread uses registers
 *    - Fewer threads per block = more registers per thread
 *    - 16×16 is a good balance
 *
 * 4. Occupancy:
 *    - Want multiple blocks per SM (streaming multiprocessor)
 *    - 16×16 allows more blocks to fit
 *    - Better utilization when some threads idle
 *
 * Common choices: 8×8, 16×16, or 32×32
 * 16×16 is popular default (good balance)
 *
 *
 * 5. Is this the fastest way to do matrix multiplication on GPU?
 * ---------------------------------------------------------------
 * No! This is a NAIVE implementation for learning. Production code uses:
 *
 * 1. Shared memory:
 *    - Cache tiles of A and B in fast shared memory
 *    - Reduces global memory accesses by 10-100x
 *    - Much more complex code
 *
 * 2. Memory coalescing optimization:
 *    - Arrange memory access patterns for optimal bandwidth
 *    - Transpose one matrix for better cache utilization
 *
 * 3. Tensor cores (modern NVIDIA):
 *    - Specialized hardware for matrix multiplication
 *    - 10-100x faster than our naive kernel
 *    - Used by cuBLAS library
 *
 * 4. Libraries like cuBLAS:
 *    - Highly optimized by NVIDIA engineers
 *    - Hand-tuned for each GPU architecture
 *    - Can be 50-100x faster than our naive version
 *
 * Performance comparison for 1024×1024 (rough estimates):
 * - Our naive kernel: ~50-100 ms
 * - Optimized with shared memory: ~10-20 ms
 * - cuBLAS (Tensor Cores): ~1-2 ms
 *
 * Our implementation is 50-100x slower than cuBLAS, but it's simple to
 * understand and demonstrates the GPU programming concepts!
 *
 * For production: ALWAYS use cuBLAS or similar libraries for matrix operations.
 * This example is for learning GPU programming fundamentals.
 *
 *
 * 6. What real applications use matrix multiplication?
 * -----------------------------------------------------
 * Everywhere in modern computing:
 *
 * 1. Machine Learning / AI:
 *    - Neural networks are mostly matrix multiplications
 *    - Forward pass: multiply weights by inputs
 *    - Backward pass: multiply gradients
 *    - Training a model: billions of matrix multiplications
 *
 * 2. Computer Graphics:
 *    - 3D transformations (rotation, scaling, translation)
 *    - Every vertex transformed by 4×4 matrices
 *    - Lighting calculations
 *    - Modern games: millions of matrix ops per frame
 *
 * 3. Scientific Computing:
 *    - Linear algebra (solving equations)
 *    - Physics simulations (forces, collisions)
 *    - Climate modeling
 *    - Quantum mechanics simulations
 *
 * 4. Image Processing:
 *    - Convolutions (can be expressed as matrix multiplication)
 *    - Filtering, transformations
 *    - Video encoding/decoding
 *
 * 5. Data Analytics:
 *    - Principal Component Analysis (PCA)
 *    - Dimensionality reduction
 *    - Recommendation systems
 *
 * Matrix multiplication is so fundamental that NVIDIA built special hardware
 * (Tensor Cores) just to do it faster. GPUs are essentially "matrix
 * multiplication machines" optimized for this specific operation.
 */
