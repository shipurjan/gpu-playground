# GPU Playground

Learning GPU programming through CUDA and WebGPU examples.

**Live WebGPU demos:** https://shipurjan.github.io/gpu-playground/

## About This Repo

This is a hands-on learning repository where I taught myself GPU computing from scratch using [Claude Code](https://claude.com/claude-code). I asked questions about GPU concepts, CUDA syntax, and parallel programming patterns, and documented the answers directly in the source code comments for future reference.

The goal was to build intuition for GPU programming by starting with simple CUDA examples, then applying those concepts to WebGPU for browser-based 3D graphics.

## Structure

```
cuda/          # CUDA examples (C/C++) - 6 examples from basics to image processing
webgpu/        # WebGPU examples (TypeScript) - 5 examples from compute to 3D graphics
```

**CUDA Examples:**
- 01-hello-gpu.cu - Basic GPU kernel execution
- 02-pointer-basics.cu - Memory management and pointers
- 03-memory-basics.cu - CPU/GPU memory operations
- 04-understanding-blocks.cu - Thread organization
- 05-matrix-multiplication.cu - Parallel matrix operations
- 06-image-blur.cu - Real-world image processing

**WebGPU Examples:**
- 01-hello-gpu - Compute shader basics
- 02-canvas-triangle - Rendering a 2D triangle
- 03-rotating-triangle - Animation with uniforms
- 04-rotating-cube - 3D wireframe with perspective projection
- 05-wgsl-reference - WGSL language reference guide

## Prerequisites

**CUDA:**
- NVIDIA GPU
- CUDA Toolkit installed ([WSL2 installation guide](INSTALLATION_WSL.md))
- C/C++ compiler

**WebGPU:**
- Modern browser with WebGPU support (Chrome 113+, Edge 113+, Safari 18+)
- Node.js 18+ for building TypeScript examples locally

## Running Examples

**CUDA:**
```bash
cd cuda
nvcc example.cu -o example
./example
```

**WebGPU:**

**View online:** https://shipurjan.github.io/gpu-playground/

**Or run locally:**
```bash
cd webgpu
npm install
npm run dev
```

## Learning Path

This repo follows a progression from low-level GPU concepts to high-level browser graphics:

1. **CUDA basics** - Parallel execution, memory management, pointers
2. **Thread organization** - Blocks, grids, workgroup sizing
3. **Real applications** - Matrix operations, image processing
4. **WebGPU fundamentals** - Compute shaders, render pipelines
5. **3D graphics** - Perspective projection, animation, WGSL shaders

All examples include detailed FAQ sections answering common questions encountered during learning.

## Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WGSL Spec](https://www.w3.org/TR/WGSL/)
