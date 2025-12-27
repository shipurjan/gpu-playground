/*
 * ============================================================================
 * KEY CONCEPTS
 * ============================================================================
 * - WGSL = WebGPU Shading Language (like GLSL or HLSL)
 * - Decorators = @attributes that tell GPU how to use variables/functions
 * - Data types = strongly typed (f32, i32, u32, vec2f, vec4f, etc.)
 * - Memory alignment = types must align to specific byte boundaries
 * - Storage buffers = can read type sizes back from GPU
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

    // ===== 2. Create output buffer =====
    // We'll write various type values from GPU
    const outputBuffer: GPUBuffer = device.createBuffer({
        size: 64,  // Enough for multiple values
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const stagingBuffer: GPUBuffer = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    // ===== 3. Load compute shader =====
    const shaderCode: string = await fetch('05-wgsl-reference.wgsl').then(r => r.text());
    const shaderModule: GPUShaderModule = device.createShaderModule({
        code: shaderCode
    });

    // ===== 4. Create pipeline =====
    const pipeline: GPUComputePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });

    // ===== 5. Create bind group =====
    const bindGroup: GPUBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{
            binding: 0,
            resource: { buffer: outputBuffer }
        }]
    });

    // ===== 6. Execute shader =====
    const commandEncoder: GPUCommandEncoder = device.createCommandEncoder();
    const passEncoder: GPUComputePassEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(1);
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, 64);
    device.queue.submit([commandEncoder.finish()]);

    // ===== 7. Read results =====
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(stagingBuffer.getMappedRange());

    // Helper to safely get value from array
    const get = (index: number): number => data[index] ?? 0;

    // ===== 8. Build reference display =====
    const output = document.getElementById('output5');
    if (!output) return;

    output.innerHTML = `
        <h3>WGSL Quick Reference</h3>

        <h4>Common Decorators</h4>
        <table>
            <tr>
                <th>Decorator</th>
                <th>Usage</th>
                <th>Description</th>
            </tr>
            <tr>
                <td><code>@group(0)</code></td>
                <td>Variables</td>
                <td>Bind group index (organizes related resources)</td>
            </tr>
            <tr>
                <td><code>@binding(0)</code></td>
                <td>Variables</td>
                <td>Binding index within a group</td>
            </tr>
            <tr>
                <td><code>@location(0)</code></td>
                <td>Vertex inputs/Fragment outputs</td>
                <td>Vertex attribute location or color attachment</td>
            </tr>
            <tr>
                <td><code>@builtin(position)</code></td>
                <td>Vertex output</td>
                <td>Clip-space position (required vertex output)</td>
            </tr>
            <tr>
                <td><code>@vertex</code></td>
                <td>Functions</td>
                <td>Marks function as vertex shader entry point</td>
            </tr>
            <tr>
                <td><code>@fragment</code></td>
                <td>Functions</td>
                <td>Marks function as fragment shader entry point</td>
            </tr>
            <tr>
                <td><code>@compute</code></td>
                <td>Functions</td>
                <td>Marks function as compute shader entry point</td>
            </tr>
            <tr>
                <td><code>@workgroup_size(x)</code></td>
                <td>Compute functions</td>
                <td>Number of threads per workgroup (e.g., 64)</td>
            </tr>
        </table>

        <h4>Scalar Types</h4>
        <table>
            <tr>
                <th>Type</th>
                <th>Size (bytes)</th>
                <th>Description</th>
                <th>Example Value</th>
            </tr>
            <tr>
                <td><code>f32</code></td>
                <td>4</td>
                <td>32-bit float</td>
                <td>${get(0).toFixed(2)}</td>
            </tr>
            <tr>
                <td><code>i32</code></td>
                <td>4</td>
                <td>32-bit signed integer</td>
                <td>${Math.floor(get(1))}</td>
            </tr>
            <tr>
                <td><code>u32</code></td>
                <td>4</td>
                <td>32-bit unsigned integer</td>
                <td>${Math.floor(get(2))}</td>
            </tr>
            <tr>
                <td><code>bool</code></td>
                <td>4*</td>
                <td>Boolean (stored as u32)</td>
                <td>true / false</td>
            </tr>
        </table>

        <h4>Vector Types</h4>
        <table>
            <tr>
                <th>Type</th>
                <th>Size (bytes)</th>
                <th>Description</th>
                <th>Example</th>
            </tr>
            <tr>
                <td><code>vec2f</code></td>
                <td>8</td>
                <td>2D float vector (x, y)</td>
                <td>vec2f(${get(3).toFixed(1)}, ${get(4).toFixed(1)})</td>
            </tr>
            <tr>
                <td><code>vec3f</code></td>
                <td>12</td>
                <td>3D float vector (x, y, z)</td>
                <td>vec3f(1.0, 2.0, 3.0)</td>
            </tr>
            <tr>
                <td><code>vec4f</code></td>
                <td>16</td>
                <td>4D float vector (x, y, z, w)</td>
                <td>vec4f(${get(5).toFixed(1)}, ${get(6).toFixed(1)}, ${get(7).toFixed(1)}, ${get(8).toFixed(1)})</td>
            </tr>
            <tr>
                <td><code>vec2i</code></td>
                <td>8</td>
                <td>2D integer vector</td>
                <td>vec2i(-5, 10)</td>
            </tr>
            <tr>
                <td><code>vec2u</code></td>
                <td>8</td>
                <td>2D unsigned vector</td>
                <td>vec2u(5, 10)</td>
            </tr>
        </table>

        <h4>Matrix Types</h4>
        <table>
            <tr>
                <th>Type</th>
                <th>Size (bytes)</th>
                <th>Description</th>
            </tr>
            <tr>
                <td><code>mat2x2f</code></td>
                <td>16*</td>
                <td>2×2 matrix (2 columns, 2 rows)</td>
            </tr>
            <tr>
                <td><code>mat3x3f</code></td>
                <td>48*</td>
                <td>3×3 matrix (3 columns, 3 rows)</td>
            </tr>
            <tr>
                <td><code>mat4x4f</code></td>
                <td>64</td>
                <td>4×4 matrix (4 columns, 4 rows)</td>
            </tr>
        </table>
        <small>* Matrices have alignment padding. Each column aligns to 16 bytes.</small>

        <h4>Variable Storage Classes</h4>
        <table>
            <tr>
                <th>Storage</th>
                <th>Usage</th>
                <th>Access</th>
            </tr>
            <tr>
                <td><code>var&lt;uniform&gt;</code></td>
                <td>Constants (read-only)</td>
                <td>All shader invocations see same value</td>
            </tr>
            <tr>
                <td><code>var&lt;storage, read&gt;</code></td>
                <td>Large read-only data</td>
                <td>GPU can read, CPU writes</td>
            </tr>
            <tr>
                <td><code>var&lt;storage, read_write&gt;</code></td>
                <td>Read/write data</td>
                <td>GPU can read and write</td>
            </tr>
            <tr>
                <td><code>var&lt;private&gt;</code></td>
                <td>Per-thread local</td>
                <td>Private to shader invocation</td>
            </tr>
            <tr>
                <td><code>var&lt;workgroup&gt;</code></td>
                <td>Shared within workgroup</td>
                <td>Threads in same workgroup can share</td>
            </tr>
        </table>

        <h4>Common Built-in Functions</h4>
        <table>
            <tr>
                <th>Function</th>
                <th>Description</th>
            </tr>
            <tr>
                <td><code>sin(x), cos(x), tan(x)</code></td>
                <td>Trigonometry (radians)</td>
            </tr>
            <tr>
                <td><code>sqrt(x), pow(x, y)</code></td>
                <td>Math operations</td>
            </tr>
            <tr>
                <td><code>abs(x), min(x, y), max(x, y)</code></td>
                <td>Value operations</td>
            </tr>
            <tr>
                <td><code>length(v), normalize(v)</code></td>
                <td>Vector operations</td>
            </tr>
            <tr>
                <td><code>dot(v1, v2), cross(v1, v2)</code></td>
                <td>Vector math</td>
            </tr>
            <tr>
                <td><code>mix(x, y, a)</code></td>
                <td>Linear interpolation (lerp)</td>
            </tr>
            <tr>
                <td><code>clamp(x, min, max)</code></td>
                <td>Constrain value to range</td>
            </tr>
        </table>

        <h4>Frequently Asked Questions</h4>
        <div class="faq-section">
            <details>
                <summary>Are vertex, fragment, and compute the only types of shaders?</summary>
                <p><strong>In WebGPU: Yes</strong>, these are the 3 shader types available.</p>
                <p><strong>In other APIs:</strong></p>
                <ul>
                    <li>OpenGL/Vulkan also have <strong>geometry shaders</strong> (process primitives) and <strong>tessellation shaders</strong> (subdivide geometry)</li>
                    <li>DirectX has similar variants</li>
                </ul>
                <p><strong>In CUDA:</strong> Only 1 type - <strong>compute kernels</strong>. No graphics pipeline, just parallel computation (<code>__global__</code> functions).</p>
            </details>

            <details>
                <summary>Is there one "real" underlying shader type, or are these genuinely different?</summary>
                <p>They're <strong>genuinely different pipeline stages</strong>, not just abstractions:</p>
                <ul>
                    <li><strong>Vertex shader</strong> - runs once per vertex (triangle corners)</li>
                    <li><strong>Fragment shader</strong> - runs once per pixel being drawn</li>
                    <li><strong>Compute shader</strong> - runs in arbitrary workgroups (no fixed stage)</li>
                </ul>
                <p>Under the hood, all compile to similar GPU machine code, but they have different inputs/outputs and run at different pipeline stages.</p>
            </details>

            <details>
                <summary>What happens if you mark a shader incorrectly (or don't mark it)?</summary>
                <p><strong>Compiler error!</strong> WGSL requires explicit entry point decorators.</p>
                <p>This will fail:</p>
                <pre><code>fn myShader() -> vec4f {  // ERROR: No @vertex or @fragment
    return vec4f(1.0, 0.0, 0.0, 1.0);
}</code></pre>
                <p>You can't use <code>@vertex</code> where <code>@fragment</code> is expected - they have different signatures and requirements. Entry points without decorators aren't callable as shaders.</p>
            </details>

            <details>
                <summary>What language is WGSL based on?</summary>
                <p><strong>Designed from scratch</strong> for WebGPU, but influenced by:</p>
                <ul>
                    <li><strong>Rust</strong> - syntax style (<code>let</code>, <code>fn</code>, type annotations like <code>: f32</code>)</li>
                    <li><strong>HLSL/GLSL</strong> - shader programming concepts (uniforms, vertex attributes, etc.)</li>
                    <li><strong>SPIR-V</strong> - compilation target (intermediate format)</li>
                </ul>
                <p>It's <strong>NOT</strong> an extension of anything - it's a new language designed for memory safety, explicit typing, and cross-platform compatibility.</p>
            </details>
        </div>

        <p><em>All values computed by GPU and read back via storage buffer</em></p>
    `;

    // Add CSS for tables
    const style = document.createElement('style');
    style.textContent = `
        #output5 table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0 20px 0;
            font-size: 14px;
        }
        #output5 th, #output5 td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        #output5 th {
            background-color: #f0f0f0;
            font-weight: bold;
        }
        #output5 code {
            background: #e8e8e8;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }
        #output5 h4 {
            margin-top: 25px;
            color: #444;
        }
        #output5 .faq-section {
            margin-top: 20px;
        }
        #output5 details {
            margin: 10px 0;
            padding: 12px;
            background: #fafafa;
            border-radius: 4px;
            border-left: 3px solid #0066cc;
        }
        #output5 summary {
            cursor: pointer;
            font-weight: bold;
            color: #333;
            user-select: none;
        }
        #output5 summary:hover {
            color: #0066cc;
        }
        #output5 details p {
            margin: 10px 0 5px 0;
            line-height: 1.5;
        }
        #output5 details ul {
            margin: 5px 0;
            padding-left: 25px;
        }
        #output5 details li {
            margin: 5px 0;
        }
        #output5 details pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            margin: 10px 0;
        }
        #output5 details pre code {
            background: transparent;
            color: inherit;
            padding: 0;
        }
    `;
    document.head.appendChild(style);

    stagingBuffer.unmap();
}

// Run the example
main().catch(err => {
    console.error('Error:', err);
    const output = document.getElementById('output5');
    if (output) {
        output.textContent = 'Error: ' + err.message;
    }
});

export {};
