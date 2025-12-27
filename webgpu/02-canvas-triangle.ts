/*
 * ============================================================================
 * KEY CONCEPTS
 * ============================================================================
 * - Render pipeline = draws pixels to screen (vs compute = calculations only)
 * - Vertex shader = runs once per vertex (triangle corner)
 * - Fragment shader = runs once per pixel being drawn
 * - Canvas = HTML element where GPU draws the output
 * - Vertex buffer = GPU memory holding triangle corner positions
 * - @vertex and @fragment = shader entry points (like @compute in example 01)
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
    const canvas = document.getElementById('gpuCanvas') as HTMLCanvasElement;
    if (!canvas) {
        throw new Error('Canvas element not found');
    }

    const context = canvas.getContext('webgpu');
    if (!context) {
        throw new Error('Failed to get WebGPU context');
    }

    // Configure canvas to draw to
    const canvasFormat: GPUTextureFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    // ===== 3. Define triangle vertices =====
    // Equilateral triangle centered at origin
    // Side length = 1, height = sqrt(3)/2 ≈ 0.866
    const h = Math.sqrt(3) / 2;
    const vertices = new Float32Array([
         0.0,    2 * h / 3,  // Top (centered)
        -0.5,   -h / 3,      // Bottom left
         0.5,   -h / 3,      // Bottom right
    ]);

    // ===== 4. Create vertex buffer (GPU memory for triangle data) =====
    const vertexBuffer: GPUBuffer = device.createBuffer({
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    // Copy vertex data from CPU to GPU
    device.queue.writeBuffer(vertexBuffer, 0, vertices);

    // ===== 5. Load shaders =====
    const shaderCode: string = await fetch('02-canvas-triangle.wgsl').then(r => r.text());
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
                arrayStride: 8,  // 2 floats * 4 bytes = 8 bytes per vertex
                attributes: [{
                    shaderLocation: 0,  // @location(0) in shader
                    offset: 0,
                    format: 'float32x2'  // 2D position (x, y)
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
            topology: 'triangle-list',  // Every 3 vertices = 1 triangle
        }
    });

    // ===== 7. Encode and submit render commands =====
    const commandEncoder: GPUCommandEncoder = device.createCommandEncoder();

    // Start render pass (like beginComputePass but for drawing)
    const renderPass: GPURenderPassEncoder = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: 'clear',
            clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },  // Dark gray background
            storeOp: 'store',
        }]
    });

    renderPass.setPipeline(pipeline);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.draw(3);  // Draw 3 vertices (1 triangle)
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);

    // Update HTML
    const output = document.getElementById('output2');
    if (output) {
        output.innerHTML = `
            <strong>Triangle rendered!</strong><br>
            <br>
            <em>You should see a red triangle on the canvas above</em>
        `;
    }
}

// Run the example
main().catch(err => {
    console.error('Error:', err);
    const output = document.getElementById('output2');
    if (output) {
        output.textContent = 'Error: ' + err.message;
    }
});

export {};
