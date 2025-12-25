# 01-hello-gpu.cu FAQ

## 1. What is the concept of device and host?

**Host** = CPU (your computer's processor)
**Device** = GPU (your graphics card)

In CUDA terminology:
- Code that runs on the CPU is "host code"
- Code that runs on the GPU is "device code"
- They have **separate physical memory** (system RAM vs VRAM)
- You must explicitly copy data between host and device

## 2. Does every CUDA/GPU program have an entrypoint running on CPU?

Yes, every CUDA program starts with `main()` on the CPU. The CPU is the "boss" that orchestrates GPU work:

1. Sets up data
2. Copies data to GPU memory
3. Launches GPU functions (called **kernels**)
4. Waits for GPU to finish
5. Copies results back from GPU

The term "GPU program" is fine, though CUDA specifically calls GPU functions **kernels**. A kernel is any function marked with `__global__` that runs on the GPU.

## 3. What is "block" and "thread per block"?

**Threads** in CUDA are like CPU threads but much more lightweight. GPUs are designed to run thousands of them simultaneously.

**Blocks** are groups of threads that execute together.

**Scale comparison:**
- **CPU**: You might have 8-16 threads running
- **GPU**: You might have 1,000 blocks × 1,024 threads = 1,024,000 threads!

**In the launch syntax `<<<1, 1>>>`:**
- First number (1) = 1 block
- Second number (1) = 1 thread per block
- **Total threads = 1 × 1 = 1 thread**

Example with more threads: `<<<4, 256>>>` = 4 blocks × 256 threads = 1,024 total threads

We'll explore blocks and threads in depth in later examples.

## 4. Can we not wait for the GPU to finish? What would the order be then?

GPU launches are **asynchronous by default**. The CPU launches the GPU work and immediately continues without waiting.

**With `cudaDeviceSynchronize()`:**
```
Hello from CPU!
Hello from GPU thread!    ← GPU finishes before continuing
Back to CPU!
```

**Without `cudaDeviceSynchronize()`:**
```
Hello from CPU!
Back to CPU!
[Program exits before GPU prints anything]
```

The CPU launches the GPU kernel and keeps running. The GPU executes in the background. Without synchronization, your program might end before the GPU finishes its work, so you'd never see the output.

**Use `cudaDeviceSynchronize()` when:**
- You need GPU results before continuing
- You want to ensure GPU work completes before program exits
- You're timing GPU operations

## 5. Where does `cudaDeviceSynchronize` come from?

It comes from the **CUDA Runtime API**, which is automatically included when you compile with `nvcc`.

You don't need to explicitly include a header - the compiler handles it. If you want to be explicit:
```c
#include <cuda_runtime.h>
```

But `nvcc` includes this by default, so basic CUDA programs work without it.

Other common CUDA Runtime API functions:
- `cudaMalloc()` - allocate GPU memory
- `cudaMemcpy()` - copy data between CPU and GPU
- `cudaFree()` - free GPU memory
- `cudaGetLastError()` - check for errors

These are all part of the CUDA Runtime, not from standard C headers like `stdio.h`.

## 6. What is nvcc and what language is this?

**nvcc** = **NVIDIA CUDA Compiler**

**The language** = **CUDA C/C++** (also just called "CUDA")

It's **C/C++ with GPU extensions**:

**Base language:**
- Standard C or C++

**Extensions added by CUDA:**
- `__global__` - function runs on GPU, callable from CPU
- `__device__` - function runs on GPU, callable from GPU only
- `__host__` - function runs on CPU (this is the default)
- `<<<blocks, threads>>>` - kernel launch syntax
- CUDA Runtime API functions (`cudaDeviceSynchronize`, `cudaMalloc`, etc.)

**How nvcc works:**
1. Takes `.cu` files (CUDA source code)
2. Separates host code (CPU) from device code (GPU)
3. Compiles CPU code with standard C++ compiler (gcc/clang)
4. Compiles GPU code into GPU machine code (PTX/SASS)
5. Links everything into one executable

So yes, it's "just C/C++" but with special GPU extensions that only `nvcc` understands. Regular `gcc` or `clang` won't understand `__global__` or `<<<>>>` syntax.
