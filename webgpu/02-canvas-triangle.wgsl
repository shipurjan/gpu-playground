// Vertex shader - runs once per vertex (3 times for triangle)
@vertex
fn vertexMain(@location(0) position: vec2f) -> @builtin(position) vec4f {
    // Convert 2D position to 4D clip space (x, y, z, w)
    // z = 0 (flat on screen), w = 1 (required for clip space)
    return vec4f(position, 0.0, 1.0);
}

// Fragment shader - runs once per pixel inside triangle
@fragment
fn fragmentMain() -> @location(0) vec4f {
    // Return red color (r, g, b, a)
    return vec4f(1.0, 0.0, 0.0, 1.0);
}
