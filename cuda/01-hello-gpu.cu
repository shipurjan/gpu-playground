#include <stdio.h>

/*
 * ============================================================================
 * KEY CONCEPTS
 * ============================================================================
 * - __global__ = function that runs on GPU
 * - <<<1, 1>>> = launch config (1 block, 1 thread)
 * - cudaDeviceSynchronize() = wait for GPU to finish
 * - Host = CPU, Device = GPU (separate memory)
 */

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
 * ============================================================================
 * FREQUENTLY ASKED QUESTIONS
 * ============================================================================
 *
 * 1. What is the concept of device and host?
 * --------------------------------------------
 * Host = CPU (your computer's processor)
 * Device = GPU (your graphics card)
 *
 * In CUDA terminology:
 * - Code that runs on the CPU is "host code"
 * - Code that runs on the GPU is "device code"
 * - They have separate physical memory (system RAM vs VRAM)
 * - You must explicitly copy data between host and device
 *
 *
 * 2. Does every CUDA/GPU program have an entrypoint running on CPU?
 * ------------------------------------------------------------------
 * Yes, every CUDA program starts with main() on the CPU. The CPU is the
 * "boss" that orchestrates GPU work:
 *
 * 1. Sets up data
 * 2. Copies data to GPU memory
 * 3. Launches GPU functions (called kernels)
 * 4. Waits for GPU to finish
 * 5. Copies results back from GPU
 *
 * The term "GPU program" is fine, though CUDA specifically calls GPU functions
 * "kernels". A kernel is any function marked with __global__ that runs on GPU.
 *
 *
 * 3. What is "block" and "thread per block"?
 * -------------------------------------------
 * Threads in CUDA are like CPU threads but much more lightweight. GPUs are
 * designed to run thousands of them simultaneously.
 *
 * Blocks are groups of threads that execute together.
 *
 * Scale comparison:
 * - CPU: You might have 8-16 threads running
 * - GPU: You might have 1,000 blocks × 1,024 threads = 1,024,000 threads!
 *
 * In the launch syntax <<<1, 1>>>:
 * - First number (1) = 1 block
 * - Second number (1) = 1 thread per block
 * - Total threads = 1 × 1 = 1 thread
 *
 * Example with more threads: <<<4, 256>>> = 4 blocks × 256 threads = 1,024 total
 *
 * We'll explore blocks and threads in depth in later examples.
 *
 *
 * 4. Can we not wait for the GPU to finish? What would the order be then?
 * ------------------------------------------------------------------------
 * GPU launches are asynchronous by default. The CPU launches the GPU work
 * and immediately continues without waiting.
 *
 * With cudaDeviceSynchronize():
 *   Hello from CPU!
 *   Hello from GPU thread!    <- GPU finishes before continuing
 *   Back to CPU!
 *
 * Without cudaDeviceSynchronize():
 *   Hello from CPU!
 *   Back to CPU!
 *   [Program exits before GPU prints anything]
 *
 * The CPU launches the GPU kernel and keeps running. The GPU executes in the
 * background. Without synchronization, your program might end before the GPU
 * finishes its work, so you'd never see the output.
 *
 * Use cudaDeviceSynchronize() when:
 * - You need GPU results before continuing
 * - You want to ensure GPU work completes before program exits
 * - You're timing GPU operations
 *
 *
 * 5. Where does cudaDeviceSynchronize come from?
 * -----------------------------------------------
 * It comes from the CUDA Runtime API, which is automatically included when
 * you compile with nvcc.
 *
 * You don't need to explicitly include a header - the compiler handles it.
 * If you want to be explicit:
 *   #include <cuda_runtime.h>
 *
 * But nvcc includes this by default, so basic CUDA programs work without it.
 *
 * Other common CUDA Runtime API functions:
 * - cudaMalloc() - allocate GPU memory
 * - cudaMemcpy() - copy data between CPU and GPU
 * - cudaFree() - free GPU memory
 * - cudaGetLastError() - check for errors
 *
 * These are all part of the CUDA Runtime, not from standard C headers like
 * stdio.h.
 *
 *
 * 6. What is nvcc and what language is this?
 * -------------------------------------------
 * nvcc = NVIDIA CUDA Compiler
 * The language = CUDA C/C++ (also just called "CUDA")
 *
 * It's C/C++ with GPU extensions:
 *
 * Base language:
 * - Standard C or C++
 *
 * Extensions added by CUDA:
 * - __global__ - function runs on GPU, callable from CPU
 * - __device__ - function runs on GPU, callable from GPU only
 * - __host__ - function runs on CPU (this is the default)
 * - <<<blocks, threads>>> - kernel launch syntax
 * - CUDA Runtime API functions (cudaDeviceSynchronize, cudaMalloc, etc.)
 *
 * How nvcc works:
 * 1. Takes .cu files (CUDA source code)
 * 2. Separates host code (CPU) from device code (GPU)
 * 3. Compiles CPU code with standard C++ compiler (gcc/clang)
 * 4. Compiles GPU code into GPU machine code (PTX/SASS)
 * 5. Links everything into one executable
 *
 * So yes, it's "just C/C++" but with special GPU extensions that only nvcc
 * understands. Regular gcc or clang won't understand __global__ or <<<>>>
 * syntax.
 *
 *
 * 7. If CUDA is a superset of C/C++, why not use it for everything?
 * ------------------------------------------------------------------
 * Several practical reasons:
 *
 * 1. NVIDIA-only:
 *    - CUDA only works on NVIDIA GPUs
 *    - Your code won't run on AMD GPUs, Intel GPUs, or machines without NVIDIA
 *    - Limits your audience significantly
 *
 * 2. Requires CUDA Toolkit:
 *    - Users must install CUDA Toolkit (large download, ~3GB)
 *    - Not pre-installed on most systems
 *    - Version compatibility issues
 *
 * 3. Compilation complexity:
 *    - Need nvcc compiler, not just gcc/clang
 *    - Slower compilation than standard C++
 *    - IDE/tooling support varies
 *
 * 4. Most programs don't benefit from GPU:
 *    - CPU is better for sequential logic, branching, small datasets
 *    - GPU overhead (memory copying) not worth it for small tasks
 *    - Example: Web server, text editor, CLI tool - GPU adds no value
 *
 * 5. Portability vs Performance trade-off:
 *    - Writing for specific hardware (NVIDIA) allows maximum optimization
 *    - Cross-platform code requires abstraction layers (slower, more complex)
 *
 * In practice: Use CUDA when you KNOW you need GPU compute and target NVIDIA.
 * Otherwise, stick to standard C/C++ for maximum compatibility.
 *
 *
 * 8. How do video games work cross-platform without CUDA?
 * ---------------------------------------------------------
 * Games use GRAPHICS APIs, not CUDA:
 *
 * Graphics APIs (for rendering):
 * - DirectX 11/12 (Windows, Xbox)
 * - Vulkan (Windows, Linux, Android - cross-vendor)
 * - Metal (macOS, iOS)
 * - OpenGL (older, cross-platform)
 *
 * These are for drawing triangles, textures, lighting - NOT general computation.
 *
 * For GPU compute within games (physics, AI, etc.):
 * - Compute Shaders (part of DirectX/Vulkan/Metal)
 * - Works across NVIDIA, AMD, Intel GPUs
 * - Less powerful than CUDA but vendor-neutral
 *
 * Why games don't use CUDA:
 * - CUDA = NVIDIA only (cuts out AMD/Intel GPU users)
 * - Graphics APIs already provide compute shaders
 * - Games need to run on consoles (PlayStation, Xbox) which don't support CUDA
 *
 * CUDA is for scientific computing, ML, data processing - not gaming graphics.
 *
 *
 * 9. What are cross-platform alternatives to CUDA?
 * --------------------------------------------------
 * Several options exist for GPU compute across vendors:
 *
 * 1. OpenCL (Open Computing Language):
 *    - Works on: NVIDIA, AMD, Intel, mobile GPUs, even CPUs
 *    - Similar concepts to CUDA (kernels, memory management)
 *    - More verbose, less optimized than CUDA on NVIDIA
 *    - Less popular than CUDA in practice
 *
 * 2. Vulkan Compute:
 *    - Part of Vulkan graphics API
 *    - Cross-vendor (NVIDIA, AMD, Intel)
 *    - Good for compute alongside graphics (games)
 *    - Lower-level, more complex than CUDA
 *
 * 3. SYCL:
 *    - Modern C++ abstraction over OpenCL
 *    - Cleaner syntax than OpenCL
 *    - Cross-vendor support
 *    - Smaller ecosystem than CUDA
 *
 * 4. HIP (Heterogeneous-compute Interface for Portability):
 *    - AMD's answer to CUDA
 *    - Very similar syntax to CUDA (can auto-convert CUDA code)
 *    - Works on AMD and NVIDIA GPUs
 *
 * 5. WebGPU:
 *    - Browser-based GPU API
 *    - Works across all GPUs (NVIDIA, AMD, Intel, mobile)
 *    - For web applications, not native apps
 *    - We'll explore this later in this repo!
 *
 * 6. Metal Compute:
 *    - Apple's GPU API (macOS, iOS)
 *    - Apple hardware only
 *
 * 7. DirectCompute:
 *    - Part of DirectX (Windows)
 *    - NVIDIA, AMD, Intel on Windows
 *
 * Performance comparison:
 * - CUDA: Fastest on NVIDIA (vendor-optimized)
 * - OpenCL/Vulkan/HIP: Slightly slower but portable
 * - Trade-off: Performance vs Portability
 *
 *
 * 10. Why not use cross-platform GPU APIs by default for all programs?
 * ---------------------------------------------------------------------
 * Good question! Here's why they're not the default:
 *
 * 1. Complexity overhead:
 *    - GPU APIs are verbose and complex
 *    - Simple "add two numbers" becomes 50+ lines of setup code
 *    - Compare:
 *      C++:     int result = a + b;
 *      OpenCL:  Create context, queue, buffers, kernel, copy data, execute, copy back
 *
 * 2. Most code doesn't benefit from GPU:
 *    - CPU is better for: control flow, branching, small data, sequential logic
 *    - GPU shines for: massive parallel data processing
 *    - Your text editor, web browser UI, database queries - CPU is fine
 *
 * 3. Copy overhead kills small tasks:
 *    - Copying data CPU → GPU takes time
 *    - Processing 100 numbers? CPU is faster (no copy overhead)
 *    - Processing 10 million numbers? GPU wins (parallelism beats copy cost)
 *
 * 4. Portability cost:
 *    - Cross-platform APIs sacrifice some performance for compatibility
 *    - If you KNOW your target (e.g., NVIDIA datacenter), CUDA is faster
 *    - If you need broad support (game on consoles), use graphics API compute
 *
 * 5. Ecosystem maturity:
 *    - CUDA: Massive ecosystem (libraries, tools, community)
 *    - OpenCL: Smaller community, less tooling
 *    - Vulkan: Complex, steep learning curve
 *
 * When to reach for GPU:
 * ✓ Image/video processing (millions of pixels)
 * ✓ Machine learning (billions of parameters)
 * ✓ Scientific simulations (massive datasets)
 * ✓ Cryptocurrency mining (repetitive calculations)
 * ✓ Matrix operations (linear algebra at scale)
 *
 * When to stick with CPU:
 * ✓ Web servers (I/O bound, not compute bound)
 * ✓ Business logic (branching, conditionals)
 * ✓ Small datasets (< thousands of elements)
 * ✓ Rapid prototyping (GPU adds complexity)
 * ✓ Maximum portability needed (embedded systems, old hardware)
 *
 * Think of GPU as a specialized tool: incredibly powerful for specific tasks,
 * overkill for everyday computing.
 *
 *
 * 11. What's the difference between GPU Compute and Graphics APIs?
 * ------------------------------------------------------------------
 * Great question! Here's the relationship:
 *
 * THE HARDWARE (bottom layer):
 * - Modern GPUs have thousands of compute cores
 * - Same physical hardware can do graphics OR general computation
 * - Think of it like a CPU: can run games, web browsers, video encoding, etc.
 *
 * HISTORICAL EVOLUTION:
 *
 * 1990s-2000s: Graphics APIs came first
 * - OpenGL, DirectX designed for 3D rendering
 * - GPUs were specialized for graphics (triangles, textures, pixels)
 * - Graphics pipeline: Vertices → Triangles → Pixels → Screen
 *
 * Mid-2000s: GPU Compute emerged
 * - CUDA (2007), OpenCL (2009)
 * - People realized: "Wait, these parallel processors can do MORE than graphics!"
 * - Term: GPGPU (General Purpose GPU computing)
 * - Unlocked GPUs for science, ML, data processing
 *
 * 2010s-now: Convergence
 * - Graphics APIs added compute shaders (general computation)
 * - GPU hardware became unified (same cores handle graphics and compute)
 * - Vulkan, DirectX 12, Metal include both graphics and compute
 *
 * KEY DIFFERENCES:
 *
 * Graphics APIs (DirectX, OpenGL, Vulkan graphics, Metal graphics):
 * - Designed for RENDERING to screen
 * - Work with graphics primitives: vertices, triangles, textures, pixels
 * - Graphics pipeline: vertex shaders → rasterization → fragment shaders
 * - Optimized for: drawing 3D scenes, lighting, textures
 * - Example: "Draw a 3D character with these textures and lighting"
 *
 * GPU Compute APIs (CUDA, OpenCL, Vulkan Compute):
 * - Designed for GENERAL parallel computation
 * - Work with raw data: arrays, matrices, buffers
 * - No graphics pipeline, just "run this function on 1 million data points"
 * - Optimized for: data processing, math, simulations
 * - Example: "Multiply two matrices" or "Train this neural network"
 *
 * SAME HARDWARE, DIFFERENT INTERFACES:
 *
 * Analogy: A smartphone
 * - Hardware: touchscreen, processor, camera
 * - Camera app: specialized interface for photos
 * - Calculator app: specialized interface for math
 * - Same hardware, different ways to use it
 *
 * GPU hardware:
 * - Graphics API: "Draw triangles, apply textures, output to screen"
 * - Compute API: "Process this data array in parallel, return results"
 *
 * ARE GRAPHICS APIS BUILT ON GPU COMPUTE?
 * No! Historically the opposite:
 * - Graphics APIs came first (GPUs were FOR graphics)
 * - GPU Compute came later (realized GPUs could do general work)
 * - Now they coexist as peers, both accessing the same hardware
 *
 * Modern Vulkan/DirectX 12/Metal include BOTH:
 * - Graphics pipeline for rendering
 * - Compute shaders for general computation
 * - Can mix: render a 3D scene (graphics) + run physics simulation (compute)
 *
 * WHAT IS "GPU COMPUTE"?
 * Yes, that's the term! Also called:
 * - GPGPU (General Purpose GPU)
 * - GPU Compute
 * - Parallel computing
 *
 * It means: Using GPU's parallel processors for NON-GRAPHICS tasks
 * - Machine learning (not rendering)
 * - Scientific simulations (not rendering)
 * - Data processing (not rendering)
 * - Cryptocurrency mining (not rendering)
 *
 * WHY SEPARATE APIS?
 *
 * 1. Different optimization goals:
 *    Graphics: Low latency, 60+ FPS, visual output
 *    Compute: Maximum throughput, accuracy, numerical results
 *
 * 2. Different workflows:
 *    Graphics: Vertices → Triangles → Pixels (fixed pipeline)
 *    Compute: Input data → Parallel function → Output data (flexible)
 *
 * 3. Different audiences:
 *    Graphics: Game developers, 3D artists, UI designers
 *    Compute: Scientists, ML engineers, data analysts
 *
 * WHICH SHOULD YOU LEARN?
 *
 * For rendering (games, 3D apps, UI):
 * → Learn graphics APIs (Vulkan, DirectX, Metal, OpenGL)
 * → This repo will cover WebGPU for browser graphics
 *
 * For data processing (ML, science, analytics):
 * → Learn compute APIs (CUDA, OpenCL, Vulkan Compute)
 * → This repo focuses on CUDA compute basics
 *
 * For games that need both:
 * → Modern graphics API with compute shaders
 * → Example: Vulkan graphics for rendering + Vulkan compute for physics
 *
 * BOTTOM LINE:
 * - GPU hardware is general-purpose parallel processors
 * - Graphics APIs expose it for rendering (specialized, came first)
 * - Compute APIs expose it for general computation (broader, came later)
 * - Modern APIs (Vulkan, DX12, Metal) include both
 * - Same hardware, different programming models for different tasks
 *
 *
 * 12. Is there a graphics API for CUDA, like Vulkan is for Vulkan Compute?
 * --------------------------------------------------------------------------
 * No! CUDA is compute-only. Here's the difference in design philosophy:
 *
 * VULKAN APPROACH (unified):
 * - Vulkan Graphics: For rendering (triangles, textures, screen output)
 * - Vulkan Compute: For general computation (data processing)
 * - One API, two modes, by Khronos Group (cross-vendor)
 *
 * NVIDIA APPROACH (separate):
 * - CUDA: For compute only (NVIDIA's proprietary compute API)
 * - For graphics: Use standard graphics APIs (DirectX, OpenGL, Vulkan)
 * - Two separate APIs, CUDA is NVIDIA-only, graphics APIs are cross-vendor
 *
 * WHY NO "CUDA GRAPHICS"?
 *
 * 1. CUDA was designed for compute, not graphics:
 *    - CUDA launched in 2007 to unlock GPU compute
 *    - Graphics was already well-served by DirectX, OpenGL
 *    - No need for NVIDIA to create another graphics API
 *
 * 2. Graphics APIs already work on NVIDIA GPUs:
 *    - DirectX, Vulkan, OpenGL all support NVIDIA hardware
 *    - These are industry standards, widely adopted
 *    - Creating a proprietary graphics API would fragment the ecosystem
 *
 * 3. Different target audiences:
 *    - CUDA: Scientists, ML engineers (need compute, not rendering)
 *    - Graphics APIs: Game developers, 3D artists (need rendering)
 *
 * CAN YOU USE CUDA + GRAPHICS TOGETHER?
 * Yes! CUDA can interoperate with graphics APIs:
 *
 * Example workflow:
 * 1. Render 3D scene with OpenGL/Vulkan/DirectX
 * 2. Process physics simulation with CUDA
 * 3. Share GPU memory between them (no CPU copy needed)
 * 4. Update graphics with CUDA results
 *
 * CUDA-Graphics interop APIs:
 * - cudaGraphicsGLRegisterBuffer() - share OpenGL buffers with CUDA
 * - cudaGraphicsD3D11RegisterResource() - share DirectX with CUDA
 * - cudaGraphicsVkRegisterImage() - share Vulkan with CUDA (newer)
 *
 * COMPARISON TABLE:
 *
 * Vulkan (unified approach):
 * ✓ One API to learn
 * ✓ Tight integration between graphics and compute
 * ✓ Cross-vendor (NVIDIA, AMD, Intel)
 * ✗ More complex, steeper learning curve
 *
 * CUDA + Graphics APIs (separate approach):
 * ✓ CUDA is simpler to learn than Vulkan Compute
 * ✓ Can pick best graphics API for your needs
 * ✓ CUDA has massive ecosystem (libraries, tools)
 * ✗ NVIDIA-only (vendor lock-in)
 * ✗ Need to learn two separate APIs
 * ✗ Interop has overhead (but optimized)
 *
 * PRACTICAL ADVICE:
 *
 * For games (graphics + compute):
 * → Use Vulkan or DirectX 12 (both have compute shaders)
 * → OR use OpenGL/DirectX + CUDA interop (if NVIDIA-only is OK)
 *
 * For scientific computing (compute only):
 * → CUDA (if NVIDIA hardware)
 * → OpenCL or Vulkan Compute (if cross-vendor needed)
 *
 * For web/browser:
 * → WebGPU (includes both graphics and compute, cross-vendor)
 *
 * BOTTOM LINE:
 * CUDA is compute-only by design. For graphics on NVIDIA GPUs, use standard
 * graphics APIs (DirectX, Vulkan, OpenGL). You can combine them via interop
 * if you need both rendering and CUDA compute in the same application.
 *
 *
 * 13. Why learn CUDA if Vulkan has compute, is cross-vendor, and open?
 * ---------------------------------------------------------------------
 * Great question! On paper, Vulkan Compute seems better. Reality is nuanced:
 *
 * VULKAN COMPUTE ADVANTAGES:
 * ✓ Cross-vendor (NVIDIA, AMD, Intel, mobile)
 * ✓ Open standard (Khronos Group, not proprietary)
 * ✓ One API for graphics + compute
 * ✓ No vendor lock-in
 *
 * CUDA ADVANTAGES:
 * ✓ Much simpler and easier to learn
 * ✓ Massive ecosystem (libraries, tools, community)
 * ✓ Better performance on NVIDIA (vendor-optimized)
 * ✓ Industry standard for ML/AI/scientific computing
 * ✓ Better debugging and profiling tools
 * ✓ 15+ years of documentation and tutorials
 *
 * THE COMPLEXITY DIFFERENCE (this is HUGE):
 *
 * CUDA "Hello World" (add two numbers):
 * ~30 lines of readable code
 *
 *   __global__ void add(int *a, int *b, int *c) {
 *       *c = *a + *b;
 *   }
 *   // Allocate, copy, launch, done
 *
 * Vulkan Compute "Hello World" (add two numbers):
 * ~300-500 lines of boilerplate
 *
 *   - Create instance
 *   - Select physical device
 *   - Create logical device
 *   - Create command pool and buffers
 *   - Allocate memory
 *   - Write shader in SPIR-V or GLSL
 *   - Create descriptor sets
 *   - Create pipeline
 *   - Submit commands
 *   - Wait for completion
 *   - Clean up 15+ objects
 *
 * This is not an exaggeration. Vulkan is EXTREMELY verbose.
 *
 * ECOSYSTEM COMPARISON:
 *
 * CUDA Libraries (all mature, widely used):
 * - cuBLAS: Linear algebra
 * - cuDNN: Deep learning primitives
 * - cuFFT: Fast Fourier transforms
 * - Thrust: C++ parallel algorithms
 * - cuSPARSE: Sparse matrix operations
 * - NCCL: Multi-GPU communication
 * - TensorRT: ML inference optimization
 *
 * Vulkan Compute ecosystem:
 * - Much smaller
 * - Fewer high-level libraries
 * - Less mature tooling
 * - Mostly used in games, not scientific computing
 *
 * REAL-WORLD ADOPTION:
 *
 * Machine Learning / AI:
 * - PyTorch: CUDA backend (primary)
 * - TensorFlow: CUDA backend (primary)
 * - JAX: CUDA backend
 * - Vulkan? Rarely used in ML
 *
 * Scientific Computing:
 * - NVIDIA dominates HPC (high-performance computing)
 * - Supercomputers use NVIDIA + CUDA
 * - Research papers assume CUDA
 *
 * Gaming:
 * - DirectX or Vulkan for graphics + compute
 * - CUDA rarely used (vendor lock-in unacceptable)
 *
 * Mobile:
 * - Vulkan Compute makes sense (cross-vendor)
 * - Metal on iOS
 *
 * IS VULKAN TRULY OPEN SOURCE?
 *
 * Vulkan specification: Open (free to implement)
 * Vulkan drivers: Closed source (NVIDIA, AMD, Intel proprietary)
 *
 * So: Open standard, closed implementations (mostly)
 * - Mesa (open-source Vulkan driver) exists but less optimized
 * - You still depend on vendor drivers for performance
 *
 * IS CUDA CLOSED SOURCE?
 *
 * Yes:
 * - CUDA Toolkit: Free to use, but closed source
 * - CUDA driver: Proprietary
 * - NVIDIA doesn't publish CUDA source code
 *
 * But:
 * - Free to download and use
 * - Permissive licensing for most use cases
 * - Not truly "closed" in practice (free tools, debuggers, profilers)
 *
 * PERFORMANCE COMPARISON:
 *
 * On NVIDIA GPUs:
 * - CUDA: Fastest (vendor-optimized, 15+ years of tuning)
 * - Vulkan Compute: Slightly slower (generic, cross-vendor compromises)
 * - Difference: 10-30% slower for compute (varies by workload)
 *
 * Why CUDA is faster on NVIDIA:
 * - Deep hardware knowledge
 * - Specialized optimizations
 * - Better memory access patterns
 * - More mature compiler
 *
 * THE VENDOR LOCK-IN QUESTION:
 *
 * Yes, CUDA locks you to NVIDIA. Is this bad?
 *
 * Consider:
 * 1. Market reality:
 *    - NVIDIA dominates AI/ML market (~95% market share)
 *    - Data centers use NVIDIA GPUs
 *    - If you're doing ML, you're probably using NVIDIA anyway
 *
 * 2. ROCm/HIP (AMD's answer):
 *    - Can auto-convert CUDA code to run on AMD
 *    - Hipify tool translates CUDA → HIP
 *    - Not perfect but helps with portability
 *
 * 3. Abstraction layers exist:
 *    - PyTorch/TensorFlow abstract GPU backend
 *    - You write Python, framework handles CUDA
 *    - Portability at application level, not kernel level
 *
 * WHEN TO CHOOSE WHAT:
 *
 * Choose CUDA if:
 * ✓ Doing ML/AI (industry standard)
 * ✓ Scientific computing on NVIDIA hardware
 * ✓ Need extensive libraries (cuDNN, cuBLAS, etc.)
 * ✓ Want simpler learning curve
 * ✓ Performance is critical on NVIDIA
 * ✓ Working in research/academia (CUDA is standard)
 *
 * Choose Vulkan Compute if:
 * ✓ Making a game (need graphics + compute)
 * ✓ Must support AMD/Intel GPUs
 * ✓ Mobile application (cross-vendor critical)
 * ✓ Already using Vulkan for graphics
 * ✓ Vendor lock-in is unacceptable for your use case
 *
 * Choose OpenCL/SYCL if:
 * ✓ Need cross-vendor compute (not graphics)
 * ✓ Don't want Vulkan's complexity
 * ✓ Enterprise requirement for vendor neutrality
 *
 * THE PRAGMATIC ANSWER:
 *
 * For learning GPU programming concepts:
 * → Start with CUDA (much easier, transfers to other APIs)
 * → Concepts translate: threads, blocks, memory, synchronization
 * → Once you understand GPU programming, Vulkan makes more sense
 *
 * For professional work:
 * → If ML/AI: CUDA (no question)
 * → If gaming: Vulkan or DirectX
 * → If mobile: Vulkan or Metal
 * → If scientific computing: CUDA (most common) or OpenCL (cross-vendor)
 *
 * BOTTOM LINE:
 *
 * Yes, Vulkan Compute is more open and cross-vendor. But:
 * - CUDA is MUCH easier to learn and use
 * - CUDA has massive ecosystem and industry adoption
 * - CUDA is faster on NVIDIA hardware
 * - Vulkan Compute is extremely verbose and complex
 * - Most ML/AI/scientific computing uses CUDA in practice
 *
 * Learn CUDA first to understand GPU programming concepts. If you later need
 * cross-vendor support, the concepts transfer to Vulkan/OpenCL/SYCL. Starting
 * with Vulkan Compute is like learning to drive on a manual transmission
 * semi-truck instead of an automatic sedan - technically more capable, but
 * much harder to learn the basics.
 *
 * This repo teaches CUDA for compute, then WebGPU for browser graphics - you'll
 * get both perspectives with reasonable learning curves.
 */
