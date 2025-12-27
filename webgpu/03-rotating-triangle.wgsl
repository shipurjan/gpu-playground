// Uniform binding - rotation angle passed from CPU each frame
@group(0) @binding(0) var<uniform> rotation: f32;

// Vertex shader - rotates each vertex around origin
@vertex
fn vertexMain(@location(0) position: vec2f) -> @builtin(position) vec4f {
    // 2D rotation matrix math:
    // x' = x*cos(θ) - y*sin(θ)
    // y' = x*sin(θ) + y*cos(θ)

    let cosTheta = cos(rotation);
    let sinTheta = sin(rotation);

    let rotatedX = position.x * cosTheta - position.y * sinTheta;
    let rotatedY = position.x * sinTheta + position.y * cosTheta;

    return vec4f(rotatedX, rotatedY, 0.0, 1.0);
}

// Fragment shader - red color (same as example 02)
@fragment
fn fragmentMain() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0);
}
