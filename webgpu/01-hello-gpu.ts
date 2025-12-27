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

            <h4 style="margin-top: 30px;">Frequently Asked Questions</h4>
            <div class="faq-section">
                <details>
                    <summary>What is a "shader"? I thought it was for graphics/shadows?</summary>
                    <p>The word "shader" is confusing because it has TWO meanings:</p>
                    <p><strong>ORIGINAL MEANING (graphics):</strong></p>
                    <ul>
                        <li>Programs that calculate lighting/shadows/colors in 3D scenes</li>
                        <li>Minecraft shaders = add realistic lighting, shadows, water effects</li>
                        <li>These are "fragment shaders" that color each pixel</li>
                    </ul>
                    <p><strong>MODERN MEANING:</strong></p>
                    <ul>
                        <li>ANY program that runs on the GPU (not just graphics!)</li>
                        <li>Confusing name, but it stuck in the industry</li>
                        <li>This file is a "compute shader" = no graphics, just computation</li>
                    </ul>
                    <p>Think of it like: <strong>"shader" = "GPU program"</strong> (broader term now).</p>
                </details>

                <details>
                    <summary>What are the different types of shaders?</summary>
                    <p><strong>GRAPHICS SHADERS (for rendering 3D scenes):</strong></p>
                    <ul>
                        <li><strong>Vertex shader</strong>: transforms 3D positions (moves vertices)</li>
                        <li><strong>Fragment shader</strong>: colors pixels (Minecraft shaders do this!)</li>
                        <li><strong>Geometry shader</strong>: generates new geometry</li>
                        <li><strong>Tessellation shaders</strong>: subdivides surfaces</li>
                    </ul>
                    <p><strong>COMPUTE SHADER (what we're using):</strong></p>
                    <ul>
                        <li>General-purpose GPU program (like CUDA kernels)</li>
                        <li>NOT for graphics - just parallel data processing</li>
                        <li>WebGPU equivalent to CUDA <code>__global__</code> functions</li>
                        <li>No vertices, no pixels, just raw computation</li>
                    </ul>
                </details>

                <details>
                    <summary>How does WGSL compare to CUDA?</summary>
                    <p><strong>CUDA (simpler syntax):</strong></p>
                    <pre><code>__global__ void myKernel(int* data) {
    data[0] = 42;
}</code></pre>
                    <p><strong>WGSL (more decorators):</strong></p>
                    <pre><code>@group(0) @binding(0) var&lt;storage, read_write&gt; data: u32;
@compute @workgroup_size(1)
fn myKernel() {
    data = 42u;
}</code></pre>
                    <p>Both do the exact same thing! WGSL is just more explicit.</p>
                </details>

                <details>
                    <summary>What do the @ decorators mean?</summary>
                    <p>The <code>@</code> syntax tells the GPU compiler how to set up the shader:</p>
                    <p><strong><code>@group(0) @binding(0)</code>:</strong></p>
                    <ul>
                        <li>Connects this variable to a buffer at binding slot 0</li>
                        <li>Like CUDA function parameters (how data gets in/out)</li>
                    </ul>
                    <p><strong><code>@compute</code>:</strong></p>
                    <ul>
                        <li>This is a compute shader (like CUDA <code>__global__</code>)</li>
                        <li>Marks the entry point for GPU execution</li>
                    </ul>
                    <p><strong><code>@workgroup_size(1)</code>:</strong></p>
                    <ul>
                        <li>Run with 1 thread (like CUDA <code>&lt;&lt;&lt;1, 1&gt;&gt;&gt;</code>)</li>
                        <li>Can be <code>(64, 1, 1)</code> for 64 threads, etc.</li>
                    </ul>
                    <p>Think of decorators as metadata: <em>"here's how to wire this up"</em>.</p>
                </details>

                <details>
                    <summary>Why is WGSL more verbose than CUDA?</summary>
                    <p><strong>WebGPU runs in BROWSERS:</strong></p>
                    <ul>
                        <li>Needs explicit security boundaries</li>
                        <li>Must work across all GPUs (NVIDIA, AMD, Intel, Apple)</li>
                        <li>Browser sandbox requires more explicit configuration</li>
                    </ul>
                    <p><strong>CUDA runs NATIVELY:</strong></p>
                    <ul>
                        <li>Direct hardware access (simpler, fewer safety checks)</li>
                        <li>NVIDIA-only (can optimize for one vendor)</li>
                        <li>No browser security concerns</li>
                    </ul>
                    <p><strong>Trade-off:</strong> WGSL is more verbose but cross-platform and secure. CUDA is simpler but NVIDIA-only and requires native installation.</p>
                </details>
            </div>
        `;
    }

    stagingBuffer.unmap();

    // Add CSS for FAQ styling
    const style = document.createElement('style');
    style.textContent = `
        #output .faq-section {
            margin-top: 20px;
        }
        #output details {
            margin: 10px 0;
            padding: 12px;
            background: #fafafa;
            border-radius: 4px;
            border-left: 3px solid #0066cc;
        }
        #output summary {
            cursor: pointer;
            font-weight: bold;
            color: #333;
            user-select: none;
        }
        #output summary:hover {
            color: #0066cc;
        }
        #output details p {
            margin: 10px 0 5px 0;
            line-height: 1.5;
        }
        #output details ul {
            margin: 5px 0;
            padding-left: 25px;
        }
        #output details li {
            margin: 5px 0;
        }
        #output details pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            margin: 10px 0;
        }
        #output details pre code {
            background: transparent;
            color: inherit;
            padding: 0;
        }
        #output code {
            background: #e8e8e8;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }
    `;
    document.head.appendChild(style);
}

// Run the example
main().catch(err => {
    console.error('Error:', err);
    const output = document.getElementById('output');
    if (output) {
        output.textContent = 'Error: ' + err.message;
    }
});

export {};
