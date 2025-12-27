// Vertex shader - simple perspective projection
@vertex
fn vertexMain(@location(0) position: vec3f) -> @builtin(position) vec4f {
    // Simple perspective: x' = x/z, y' = y/z
    // Input position is already rotated and translated (done on CPU)

    let x = position.x / position.z;
    let y = position.y / position.z;

    // Map to clip space (-1 to 1 for x,y; 0 to 1 for z)
    // Z for depth testing: map to 0-1 range
    let z = (position.z - 0.5) / 1.5;  // Normalize depth

    return vec4f(x, y, z, 1.0);
}

// Fragment shader - solid red color (same as triangles)
@fragment
fn fragmentMain() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0);  // Red wireframe
}
