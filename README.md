# GPU Playground

Learning GPU programming through CUDA and WebGPU examples.

## Structure


```
cuda/          # CUDA examples (C/C++)
webgpu/        # WebGPU examples (TypeScript/JavaScript)
```

## Prerequisites

**CUDA:**
- NVIDIA GPU
- CUDA Toolkit installed ([WSL2 installation guide](INSTALLATION_WSL.md))
- C/C++ compiler

**WebGPU:**
- Modern browser (Chrome 113+, Firefox with flag)
- Node.js for TypeScript examples

## Running Examples

**CUDA:**
```bash
cd cuda
nvcc example.cu -o example
./example
```

**WebGPU:**
```bash

cd webgpu
npm install
npm run dev

```

## Learning Path


1. CUDA basics (parallel execution, memory management)
2. Thread organization (blocks, grids)
3. Real examples (matrix operations, image processing)
4. WebGPU translation (graphics shaders, compute shaders)

## Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WGSL Spec](https://www.w3.org/TR/WGSL/)
