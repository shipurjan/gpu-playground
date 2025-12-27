/*
 * ============================================================================
 * KEY CONCEPTS
 * ============================================================================
 * - Simple perspective projection: x' = x/z, y' = y/z
 * - CPU does rotation/translation, GPU does projection
 * - 8 vertices, 12 edges for wireframe cube
 * - Line-list topology renders edges directly
 */

interface Vec3 {
    x: number;
    y: number;
    z: number;
}

// Cube vertices (8 corners)
const baseVertices: Vec3[] = [
    { x:  0.25, y:  0.25, z:  0.25 },  // 0: front top right
    { x: -0.25, y:  0.25, z:  0.25 },  // 1: front top left
    { x: -0.25, y: -0.25, z:  0.25 },  // 2: front bottom left
    { x:  0.25, y: -0.25, z:  0.25 },  // 3: front bottom right
    { x:  0.25, y:  0.25, z: -0.25 },  // 4: back top right
    { x: -0.25, y:  0.25, z: -0.25 },  // 5: back top left
    { x: -0.25, y: -0.25, z: -0.25 },  // 6: back bottom left
    { x:  0.25, y: -0.25, z: -0.25 },  // 7: back bottom right
];

// Edge indices (12 edges, 2 vertices each)
const edges = [
    0, 1,  1, 2,  2, 3,  3, 0,  // Front face
    4, 5,  5, 6,  6, 7,  7, 4,  // Back face
    0, 4,  1, 5,  2, 6,  3, 7,  // Connecting edges
];

function rotateXZ(v: Vec3, angle: number): Vec3 {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return {
        x: v.x * c - v.z * s,
        y: v.y,
        z: v.x * s + v.z * c,
    };
}

function translateZ(v: Vec3, dz: number): Vec3 {
    return { x: v.x, y: v.y, z: v.z + dz };
}

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
    const canvas = document.getElementById('gpuCanvas3') as HTMLCanvasElement;
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

    // ===== 3. Create vertex buffer (will update each frame) =====
    const vertexBuffer: GPUBuffer = device.createBuffer({
        size: 8 * 3 * 4,  // 8 vertices * 3 floats (x,y,z) * 4 bytes
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    // ===== 4. Create index buffer =====
    const indexBuffer: GPUBuffer = device.createBuffer({
        size: edges.length * 2,  // 24 indices * 2 bytes (Uint16)
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indexBuffer, 0, new Uint16Array(edges));

    // ===== 5. Load shaders =====
    const shaderCode: string = await fetch('04-rotating-cube.wgsl').then(r => r.text());
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
                arrayStride: 12,  // 3 floats * 4 bytes
                attributes: [{
                    shaderLocation: 0,
                    offset: 0,
                    format: 'float32x3'  // vec3f position
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
            topology: 'line-list',
        },
    });

    // ===== 7. Animation loop =====
    let startTime = Date.now();

    function render(): void {
        if (!context) return;

        const elapsed = (Date.now() - startTime) / 1000;
        const angle = elapsed * Math.PI * 0.5;  // Rotate at π/2 radians per second
        const dz = 1.0;  // Distance from camera

        // Transform vertices on CPU
        const transformedVertices = new Float32Array(8 * 3);
        for (let i = 0; i < baseVertices.length; i++) {
            const baseVertex = baseVertices[i];
            if (baseVertex) {
                const v = translateZ(rotateXZ(baseVertex, angle), dz);
                transformedVertices[i * 3 + 0] = v.x;
                transformedVertices[i * 3 + 1] = v.y;
                transformedVertices[i * 3 + 2] = v.z;
            }
        }

        // Update vertex buffer
        device.queue.writeBuffer(vertexBuffer, 0, transformedVertices);

        // Render
        const commandEncoder: GPUCommandEncoder = device.createCommandEncoder();
        const renderPass: GPURenderPassEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: 'clear',
                clearValue: { r: 0.06, g: 0.06, b: 0.06, a: 1.0 },  // #101010
                storeOp: 'store',
            }]
        });

        renderPass.setPipeline(pipeline);
        renderPass.setVertexBuffer(0, vertexBuffer);
        renderPass.setIndexBuffer(indexBuffer, 'uint16');
        renderPass.drawIndexed(edges.length);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(render);
    }

    render();

    // Update HTML
    const output = document.getElementById('output4');
    if (output) {
        output.innerHTML = `
            <strong>Wireframe cube rotating!</strong><br>
            <br>
            <em>Simple perspective: x' = x/z, y' = y/z</em><br>
            <br>
            <small>Based on: <a href="https://www.youtube.com/watch?v=qjWkNZ0SXfo" target="_blank">One Formula That Demystifies 3D Graphics</a></small>
        `;
    }
}

// Run the example
main().catch(err => {
    console.error('Error:', err);
    const output = document.getElementById('output4');
    if (output) {
        output.textContent = 'Error: ' + err.message;
    }
});

export {};
