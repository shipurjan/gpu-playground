// Storage buffer to write example values
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main() {
    // Demonstrate various WGSL types by writing example values

    // Scalar types
    output[0] = 3.14159;           // f32 - 32-bit float
    output[1] = f32(-42);          // i32 stored as f32
    output[2] = f32(255);          // u32 stored as f32

    // Vector type examples
    let v2 = vec2f(1.0, 2.0);      // 2D vector
    output[3] = v2.x;
    output[4] = v2.y;

    let v4 = vec4f(1.0, 0.5, 0.25, 1.0);  // 4D vector (like RGBA color)
    output[5] = v4.x;
    output[6] = v4.y;
    output[7] = v4.z;
    output[8] = v4.w;

    // Demonstrate some built-in functions
    let angle = 1.57;  // ~90 degrees in radians
    output[9] = sin(angle);        // Should be ~1.0
    output[10] = cos(angle);       // Should be ~0.0

    let vec_a = vec2f(3.0, 4.0);
    output[11] = length(vec_a);    // Should be 5.0 (Pythagorean theorem)

    // Vector operations
    let vec_b = normalize(vec_a);  // Make length = 1.0
    output[12] = vec_b.x;          // Should be 0.6
    output[13] = vec_b.y;          // Should be 0.8

    // Math functions
    output[14] = sqrt(16.0);       // 4.0
    output[15] = pow(2.0, 3.0);    // 8.0
}
