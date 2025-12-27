/*
 * ============================================================================
 * KEY CONCEPTS
 * ============================================================================
 * - Uniform buffer = data passed from CPU to GPU each frame (like rotation angle)
 * - requestAnimationFrame = browser API for smooth 60fps animation
 * - Continuous rendering = GPU renders every frame (this is where GPU wins)
 * - Rotation matrix = mathematical transform to rotate vertices
 * - writeBuffer = update GPU memory without recreating buffers
 */

async function main(): Promise<void> {
    // ===== 1. Initialize WebGPU =====
    if (!navigator.gpu) {
        throw new Error('WebGPU not supported in this browser');
    }

    const adapter: GPUAdapter | null = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error('Failed to get GPU adapter');
    }

    const device: GPUDevice = await adapter.requestDevice();

    // ===== 2. Set up canvas =====
    const canvas = document.getElementById('gpuCanvas2') as HTMLCanvasElement;
    if (!canvas) {
        throw new Error('Canvas element not found');
    }

    const context = canvas.getContext('webgpu');
    if (!context) {
        throw new Error('Failed to get WebGPU context');
    }

    const canvasFormat: GPUTextureFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    // ===== 3. Define triangle vertices =====
    // Equilateral triangle centered at origin (rotates around center)
    // Side length = 1, height = sqrt(3)/2 ≈ 0.866
    const h = Math.sqrt(3) / 2;
    const vertices = new Float32Array([
         0.0,    2 * h / 3,  // Top (centered)
        -0.5,   -h / 3,      // Bottom left
         0.5,   -h / 3,      // Bottom right
    ]);

    const vertexBuffer: GPUBuffer = device.createBuffer({
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, 0, vertices);

    // ===== 4. Create uniform buffer for rotation angle =====
    // Uniform = constant value available to all shader invocations
    const uniformBuffer: GPUBuffer = device.createBuffer({
        size: 4,  // 1 float (4 bytes) for rotation angle
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ===== 5. Load shaders =====
    const shaderCode: string = await fetch('03-rotating-triangle.wgsl').then(r => r.text());
    const shaderModule: GPUShaderModule = device.createShaderModule({
        code: shaderCode
    });

    // ===== 6. Create render pipeline =====
    const pipeline: GPURenderPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
            buffers: [{
                arrayStride: 8,
                attributes: [{
                    shaderLocation: 0,
                    offset: 0,
                    format: 'float32x2'
                }]
            }]
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{
                format: canvasFormat
            }]
        },
        primitive: {
            topology: 'triangle-list',
        }
    });

    // ===== 7. Create bind group for uniform =====
    const bindGroup: GPUBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }]
    });

    // ===== 8. Animation loop =====
    let startTime = Date.now();

    function render(): void {
        if (!context) return;  // Safety check for TypeScript

        // Calculate rotation angle based on elapsed time
        const elapsed = (Date.now() - startTime) / 1000;  // seconds
        const rotation = elapsed;  // radians per second

        // Update uniform buffer with new rotation angle
        device.queue.writeBuffer(
            uniformBuffer,
            0,
            new Float32Array([rotation])
        );

        // Encode render commands
        const commandEncoder: GPUCommandEncoder = device.createCommandEncoder();

        const renderPass: GPURenderPassEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: 'clear',
                clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
                storeOp: 'store',
            }]
        });

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.setVertexBuffer(0, vertexBuffer);
        renderPass.draw(3);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);

        // Request next frame (60fps loop)
        requestAnimationFrame(render);
    }

    // Start animation
    render();

    // Update HTML
    const output = document.getElementById('output3');
    if (output) {
        output.innerHTML = `
            <strong>Triangle rotating!</strong><br>
            <br>
            <em>GPU renders this 60 times per second - this is where GPU wins</em>
        `;
    }
}

// Run the example
main().catch(err => {
    console.error('Error:', err);
    const output = document.getElementById('output3');
    if (output) {
        output.textContent = 'Error: ' + err.message;
    }
});

export {};
