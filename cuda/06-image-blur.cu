#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Suppress warnings from stb_image library
#pragma nv_diag_suppress 550

// stb_image library for loading/saving images
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#pragma nv_diag_default 550

/*
 * ============================================================================
 * KEY CONCEPTS
 * ============================================================================
 * - Images stored as 1D arrays in row-major order: image[y * width + x]
 * - Convolution: applying filter kernel across image
 * - Box blur: average pixels in NxN neighborhood
 * - Each pixel independent (perfect for GPU parallelism)
 * - Boundary handling: clamp to edges (skip out-of-bounds pixels)
 * - Thread maps to pixel: (x,y) = (blockIdx * blockDim + threadIdx)
 * - h_ prefix = host (CPU) memory, d_ prefix = device (GPU) memory
 */

// GPU box blur kernel for RGB images
__global__ void blurKernel(float* d_input, float* d_output, int width, int height, int radius) {
    // Calculate global pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (x >= width || y >= height) return;

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    int count = 0;

    // Average neighborhood pixels for each RGB channel
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            // Boundary check: only include pixels within image
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int idx = (ny * width + nx) * 3;
                sum_r += d_input[idx];
                sum_g += d_input[idx + 1];
                sum_b += d_input[idx + 2];
                count++;
            }
        }
    }

    int out_idx = (y * width + x) * 3;
    d_output[out_idx] = sum_r / count;
    d_output[out_idx + 1] = sum_g / count;
    d_output[out_idx + 2] = sum_b / count;
}

// Load image from disk as RGB floats (0.0 - 1.0)
float* loadImage(const char* filename, int* width, int* height) {
    int channels;
    unsigned char* img = stbi_load(filename, width, height, &channels, 3);  // Force 3 channels

    if (!img) {
        printf("Error: Failed to load %s\n", filename);
        return NULL;
    }

    printf("Loaded %s: %dx%d\n", filename, *width, *height);

    // Convert to RGB float array
    int size = (*width) * (*height) * 3;
    float* h_rgb = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        h_rgb[i] = img[i] / 255.0f;
    }

    stbi_image_free(img);
    return h_rgb;
}

// Save RGB float array to PNG file
void saveImage(const char* filename, float* h_image, int width, int height) {
    int size = width * height * 3;
    unsigned char* img = (unsigned char*)malloc(size);

    for (int i = 0; i < size; i++) {
        // Clamp and convert to byte
        float val = fmaxf(0.0f, fminf(1.0f, h_image[i]));
        img[i] = (unsigned char)(val * 255.0f);
    }

    if (stbi_write_png(filename, width, height, 3, img, width * 3)) {
        printf("Saved %s\n", filename);
    } else {
        printf("Error: Failed to save %s\n", filename);
    }

    free(img);
}

int main() {
    const char* input_filename = "cat.jpg";
    const char* output_filename = "cat_blurred.png";
    const int blur_radius = 5;

    printf("=== GPU Image Blur ===\n");
    printf("Input: %s\n", input_filename);
    printf("Blur radius: %d (%dx%d kernel)\n\n",
           blur_radius, 2*blur_radius+1, 2*blur_radius+1);

    // Load image
    int width, height;
    float* h_input = loadImage(input_filename, &width, &height);
    if (!h_input) return 1;

    int num_pixels = width * height;
    size_t bytes = num_pixels * 3 * sizeof(float);  // RGB = 3 channels

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy input image to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Configure kernel launch
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);

    printf("Launching kernel: %dx%d blocks, %dx%d threads per block\n",
           grid_size.x, grid_size.y, block_size.x, block_size.y);

    // Launch kernel
    blurKernel<<<grid_size, block_size>>>(d_input, d_output, width, height, blur_radius);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    float* h_output = (float*)malloc(bytes);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Save result
    saveImage(output_filename, h_output, width, height);

    printf("\nProcessed %d pixels using %d GPU threads\n",
           num_pixels, grid_size.x * grid_size.y * block_size.x * block_size.y);

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

/*
 * ============================================================================
 * FREQUENTLY ASKED QUESTIONS
 * ============================================================================
 *
 * 1. Why are images stored as 1D arrays instead of 2D?
 * ------------------------------------------------------
 * GPU memory is linear (1D). We use row-major indexing: index = y * width + x
 *
 * Example 4x3 image:
 *   Conceptual 2D:     Memory layout:
 *   [0][1][2][3]       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
 *   [4][5][6][7]
 *   [8][9][10][11]
 *
 * Benefits:
 * - Memory coalescing: adjacent threads read adjacent memory (fast)
 * - Better cache utilization
 * - Standard layout for GPU image processing
 *
 *
 * 2. What is convolution and box blur?
 * -------------------------------------
 * Convolution: Sliding a filter kernel over an image, computing weighted sum
 * of neighborhood pixels.
 *
 * Box blur: Simple convolution where all weights are equal (averaging).
 *
 * For radius=1 (3x3 kernel):
 *   output[x,y] = average of 9 pixels centered at (x,y)
 *
 * Used for: smoothing, noise reduction, preparing for other operations.
 *
 *
 * 3. How does boundary handling work?
 * ------------------------------------
 * Pixels near edges have incomplete neighborhoods. We use "clamp" method:
 * - Skip out-of-bounds pixels
 * - Only average valid pixels
 * - Edge pixels have smaller neighborhoods (slightly dimmer in blur)
 *
 * Alternative methods:
 * - Wrap: out-of-bounds wraps to opposite side
 * - Mirror: reflect at boundaries
 * - Constant: use fixed value (e.g., 0)
 *
 *
 * 4. How does RGB blur work?
 * ---------------------------
 * We process all 3 color channels (Red, Green, Blue) separately.
 *
 * Memory layout (interleaved RGB):
 *   [R0, G0, B0, R1, G1, B1, R2, G2, B2, ...]
 *
 * Each pixel blur:
 *   - Average red channel from neighborhood
 *   - Average green channel from neighborhood
 *   - Average blue channel from neighborhood
 *
 * Same blur applied to each channel independently.
 *
 *
 * 5. What are h_ and d_ prefixes?
 * --------------------------------
 * Standard naming convention in CUDA:
 * - h_name = host (CPU) memory
 * - d_name = device (GPU) memory
 *
 * Example:
 *   float* h_input;   // CPU memory
 *   float* d_input;   // GPU memory
 *   cudaMalloc(&d_input, size);
 *   cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
 *
 * Benefits:
 * - Immediately clear which memory space
 * - Avoid bugs (can't dereference d_ pointers from CPU code)
 * - Standard in production CUDA code
 *
 *
 * 6. How could this be optimized?
 * --------------------------------
 * This is a naive implementation. Production optimizations:
 *
 * 1. Separable filters (5-10x faster):
 *    Box blur is separable: 2D blur = 1D horizontal + 1D vertical
 *    11x11 kernel: 121 operations → 22 operations
 *
 * 2. Shared memory (10-20x faster):
 *    Cache neighborhood in fast shared memory
 *    Reduces global memory reads
 *
 * 3. Texture memory:
 *    Hardware-accelerated interpolation and caching
 *    Automatic boundary handling
 *
 * For production: use cuDNN, NPP, or OpenCV GPU modules (50-100x faster).
 */
