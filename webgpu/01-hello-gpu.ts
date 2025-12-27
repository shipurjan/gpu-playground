/*
 * ============================================================================
 * KEY CONCEPTS
 * ============================================================================
 * - WebGPU runs in browser (JavaScript/TypeScript), CUDA runs native (C++)
 * - Shaders written in WGSL (WebGPU Shading Language), not C
 * - Everything is async (await/Promise based)
 * - No printf from GPU - must write to buffer and read back
 * - @compute = shader runs on GPU (like CUDA's __global__)
 * - @workgroup_size(1) = equivalent to CUDA's <<<1, 1>>>
 * - Cross-platform: works on NVIDIA, AMD, Intel, Apple GPUs
 */

async function main(): Promise<void> {
    // ===== 1. Initialize WebGPU (setup adapter and device) =====
    // Adapter = physical GPU, Device = logical connection to GPU
    if (!navigator.gpu) {
        throw new Error('WebGPU not supported in this browser');
    }

    const adapter: GPUAdapter | null = await navigator.gpu.requestAdapter();

    if (!adapter) {
        throw new Error('Failed to get GPU adapter');
    }

    const device: GPUDevice = await adapter.requestDevice();

    // ===== 2. Create GPU buffer for output =====
    // GPU can't printf - we write a value to a buffer instead
    // STORAGE = GPU can read/write, COPY_SRC = can copy from this buffer
    const outputBuffer: GPUBuffer = device.createBuffer({
        size: 4,  // 4 bytes = 1 u32 value
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Buffer for reading back to CPU (MAP_READ requires separate staging buffer)
    const stagingBuffer: GPUBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    // ===== 3. Load compute shader from file =====
    // WGSL (WebGPU Shading Language) - similar to CUDA but different syntax
    // Fetch shader code from separate .wgsl file
    const shaderCode: string = await fetch('01-hello-gpu.wgsl').then(r => r.text());

    const shaderModule: GPUShaderModule = device.createShaderModule({
        code: shaderCode
    });

    // ===== 4. Create pipeline (like compiling CUDA kernel) =====
    const pipeline: GPUComputePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main'  // Function name in shader
        }
    });

    // ===== 5. Create bind group (connect buffer to shader) =====
    // Tells shader: "output variable" uses "outputBuffer"
    const bindGroup: GPUBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{
            binding: 0,
            resource: { buffer: outputBuffer }
        }]
    });

    // ===== 6. Encode and submit GPU commands =====
    const commandEncoder: GPUCommandEncoder = device.createCommandEncoder();

    // Start compute pass (run shader)
    const passEncoder: GPUComputePassEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(1);  // Launch 1 workgroup (like CUDA <<<1, ...>>>)
    passEncoder.end();

    // Copy result from GPU output buffer to CPU-readable staging buffer
    commandEncoder.copyBufferToBuffer(
        outputBuffer, 0,  // source
        stagingBuffer, 0, // destination
        4                 // size in bytes
    );

    // Submit commands to GPU
    device.queue.submit([commandEncoder.finish()]);

    // ===== 7. Read result back from GPU =====
    // Wait for GPU to finish (like cudaDeviceSynchronize())
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const data: Uint32Array = new Uint32Array(stagingBuffer.getMappedRange());

    // Print result
    console.log('Hello from CPU!');
    console.log('Value from GPU:', data[0]);
    console.log('Back to CPU!');

    // Update HTML
    const output = document.getElementById('output');
    if (output) {
        output.innerHTML = `
            <strong>Hello from GPU!</strong><br>
            GPU wrote value: ${data[0]}<br>
            <br>
            <em>Check browser console for full output</em>
        `;
    }

    stagingBuffer.unmap();
}

// Run the example
main().catch(err => {
    console.error('Error:', err);
    const output = document.getElementById('output');
    if (output) {
        output.textContent = 'Error: ' + err.message;
    }
});
