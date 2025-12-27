/*
 * ============================================================================
 * KEY CONCEPTS
 * ============================================================================
 * - This is a COMPUTE SHADER (runs on GPU, not for graphics)
 * - @compute = function runs on GPU (like CUDA __global__)
 * - @workgroup_size(1) = launch with 1 thread (like CUDA <<<1, 1>>>)
 * - @group/@binding = connects variables to buffers (like CUDA parameters)
 * - WGSL is more verbose than CUDA but does same thing
 */

// Connect to buffer at binding slot 0
// var<storage, read_write> = GPU can read and write this buffer
@group(0) @binding(0) var<storage, read_write> output: u32;

// @compute = this is a compute shader (like CUDA __global__)
// @workgroup_size(1) = run with 1 thread (equivalent to CUDA <<<1, 1>>>)
@compute @workgroup_size(1)
fn main() {
    // This code runs on the GPU!
    output = 42u;  // Write the answer to everything
}

/*
 * ============================================================================
 * FREQUENTLY ASKED QUESTIONS
 * ============================================================================
 *
 * 1. What is a "shader"? I thought it was for graphics/shadows?
 * ---------------------------------------------------------------
 * The word "shader" is confusing because it has TWO meanings:
 *
 * ORIGINAL MEANING (graphics):
 * - Programs that calculate lighting/shadows/colors in 3D scenes
 * - Minecraft shaders = add realistic lighting, shadows, water effects
 * - These are "fragment shaders" that color each pixel
 *
 * MODERN MEANING:
 * - ANY program that runs on the GPU (not just graphics!)
 * - Confusing name, but it stuck in the industry
 * - This file is a "compute shader" = no graphics, just computation
 *
 * Think of it like: "shader" = "GPU program" (broader term now).
 *
 *
 * 2. What are the different types of shaders?
 * --------------------------------------------
 * GRAPHICS SHADERS (for rendering 3D scenes):
 * - Vertex shader: transforms 3D positions (moves vertices)
 * - Fragment shader: colors pixels (Minecraft shaders do this!)
 * - Geometry shader: generates new geometry
 * - Tessellation shaders: subdivides surfaces
 *
 * COMPUTE SHADER (what we're using):
 * - General-purpose GPU program (like CUDA kernels)
 * - NOT for graphics - just parallel data processing
 * - WebGPU equivalent to CUDA __global__ functions
 * - No vertices, no pixels, just raw computation
 *
 *
 * 3. How does WGSL compare to CUDA?
 * -----------------------------------
 * CUDA (simpler syntax):
 *   __global__ void myKernel(int* data) {
 *       data[0] = 42;
 *   }
 *
 * WGSL (more decorators):
 *   @group(0) @binding(0) var<storage, read_write> data: u32;
 *   @compute @workgroup_size(1)
 *   fn myKernel() {
 *       data = 42u;
 *   }
 *
 * Both do the exact same thing! WGSL is just more explicit.
 *
 *
 * 4. What do the @ decorators mean?
 * ----------------------------------
 * The @ syntax tells the GPU compiler how to set up the shader:
 *
 * @group(0) @binding(0):
 * - Connects this variable to a buffer at binding slot 0
 * - Like CUDA function parameters (how data gets in/out)
 *
 * @compute:
 * - This is a compute shader (like CUDA __global__)
 * - Marks the entry point for GPU execution
 *
 * @workgroup_size(1):
 * - Run with 1 thread (like CUDA <<<1, 1>>>)
 * - Can be (64, 1, 1) for 64 threads, etc.
 *
 * Think of decorators as metadata: "here's how to wire this up".
 *
 *
 * 5. Why is WGSL more verbose than CUDA?
 * ---------------------------------------
 * WebGPU runs in BROWSERS:
 * - Needs explicit security boundaries
 * - Must work across all GPUs (NVIDIA, AMD, Intel, Apple)
 * - Browser sandbox requires more explicit configuration
 *
 * CUDA runs NATIVELY:
 * - Direct hardware access (simpler, fewer safety checks)
 * - NVIDIA-only (can optimize for one vendor)
 * - No browser security concerns
 *
 * Trade-off: WGSL is more verbose but cross-platform and secure.
 * CUDA is simpler but NVIDIA-only and requires native installation.
 */
