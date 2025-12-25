## Repository Context

This is a learning repository where the user is teaching themselves GPU computing from scratch. The user is an experienced fullstack developer (React/TypeScript/Node.js) but is **new to GPU programming and C**.

## Learning Goals

1. **Understand GPU fundamentals** through CUDA (NVIDIA's C/C++ extension)
2. **Learn parallel computing concepts** (threads, blocks, memory management)

3. **Apply knowledge to WebGPU** for 3D graphics in the browser
4. **Build intuition** for when/why to use GPU vs CPU

## User's Background


- **Strong in**: TypeScript, JavaScript, React, Node.js
- **Learning**: C, GPU programming, parallel computing
- **Unfamiliar with**: CUDA syntax, GPU memory models, shader languages

- **Communication style**: Direct, evidence-based, minimal fluff
- **Needs**: Clear explanations, working examples, answers to "why" questions

## File Structure & Naming

### CUDA Examples (cuda/)
```
01-single-thread.cu
02-parallel-add.cu
03-thread-indexing.cu
...
```

### WebGPU Examples (webgpu/)
```
01-triangle.html
02-compute-shader.ts
...
```

**Naming convention**: `<number>-<descriptive-name>.<extension>`

**Documentation**: FAQs are documented in comment blocks at the bottom of each source file

## FAQ Documentation Format

When the user asks questions about a specific example, **add a FAQ section at the bottom of the source file** in a comment block:

```c
/*
 * ============================================================================
 * FREQUENTLY ASKED QUESTIONS
 * ============================================================================
 *
 * 1. Why is the function void instead of returning a value?
 * ----------------------------------------------------------
 * In C, when working with GPU memory, you can't return GPU pointers directly.
 * Instead, you pass an output pointer as a parameter and modify it in place.
 * This is standard C practice for avoiding memory allocation issues.
 *
 *
 * 2. What does <<<1, 8>>> mean?
 * ------------------------------
 * This is CUDA's kernel launch syntax:
 * - First number (1) = number of blocks
 * - Second number (8) = threads per block
 * - Total threads = 1 × 8 = 8 threads
 *
 *
 * 3. Why do we need to copy memory between CPU and GPU?
 * ------------------------------------------------------
 * The CPU and GPU have separate physical RAM. CPU uses system RAM, GPU uses
 * VRAM. They can't access each other's memory directly, so we must explicitly
 * copy data between them.
 */
```

### FAQ Guidelines

**IMPORTANT: Be Proactive About Documentation**

After answering user questions, **actively check if the conversation should be documented**:

1. **Review existing FAQ section** - Check if this topic is already covered in the source file
2. **Assess if it's documentable** - Did the user express confusion or uncertainty about a concept?
3. **Offer to document it** - Don't wait for the user to ask. Proactively say: "Should I add this to the FAQ section?"
4. **Think like a beginner** - Would someone new to GPU programming find this confusing? If yes, document it.

**What to document:**
- ✅ Concepts the user actually asked about or was confused by
- ✅ Syntax questions ("`__global__` means what?")
- ✅ "Why" questions (why CPU/GPU memory separation?)
- ✅ Installation issues or gotchas they encountered
- ❌ Concepts the user didn't ask about (don't add extra noise)
- ❌ Things that are obvious to the user already

**The user may forget to ask for documentation** - be proactive and suggest it when appropriate.

**When user asks questions:**
1. **Extract the essence** - rewrite question concisely to get to the core concept
2. **Answer directly** - no fluff, just the answer
3. **Add to source file's FAQ section** - append new questions to the comment block
4. **Number sequentially** - each question gets a number
5. **Split multiple questions** - if user asks 3 things at once, create 3 separate numbered entries

**Example transformation:**

User asks:
> "This is quite difficult I think. I mean easy but difficult for newcomers compared to just C or whatnot. It's like you have to already know C to get into this and then learn a few more concepts special for GPU"

Add to FAQ as:
```markdown
## 4. Why is GPU programming harder than regular C?

GPU programming assumes you already know C, then adds several new concepts:
- Manual memory management (CPU vs GPU memory)
- Explicit data copying (can't access GPU memory from CPU)
- Parallel thinking (thousands of threads instead of sequential loops)
- New syntax (`<<<>>>`, `__global__`, etc.)


It's not designed for beginners - it's for developers who need performance optimization.
```


## How to Help


### When Writing Code Examples


1. **Start simple, build complexity gradually**
   - Level 1: Single thread operations
   - Level 2: Multiple threads, basic indexing
   - Level 3: Blocks and grids
   - Level 4: Real applications (matrix multiplication, image processing)

2. **Include comments explaining GPU-specific concepts**
   ```cuda
   // threadIdx.x = unique ID for this thread (0, 1, 2...)
   int i = threadIdx.x;
   ```


3. **Show both CPU and GPU memory clearly**
   ```cuda
   int h_value = 5;    // h_ = host (CPU)
   int* d_value;       // d_ = device (GPU)
   ```

4. **Provide compile/run instructions**
   ```bash
   nvcc example.cu -o example
   ./example
   ```

### When Answering Questions

The user will ask **many questions** about:
- Syntax (`__global__`, `<<<blocks, threads>>>`)
- Memory management (why copy CPU ↔ GPU?)
- Parallel concepts (how do threads work?)

- Terminology (kernel, host, device, workgroup)
- C language features (if rusty)

**Response style:**
- ✅ Direct, concise explanations
- ✅ Use analogies when helpful

- ✅ Show code examples
- ✅ Explain "why" not just "what"
- ✅ **Always update the FAQ section in the source file**
- ❌ No excessive enthusiasm or fluff
- ❌ No "great question!" preambles

## Progression Path

### Phase 1: CUDA Fundamentals (cuda/)
1. Single thread basics
2. Parallel array operations
3. Thread indexing (`threadIdx.x`)

4. Blocks and grids (`blockIdx.x`, `blockDim.x`)

5. 2D indexing for matrices
6. Real example: Matrix multiplication
7. Memory optimization basics

### Phase 2: WebGPU Translation (webgpu/)
1. Simple triangle rendering
2. Vertex and fragment shaders
3. Compute shaders (similar to CUDA)
4. Buffer management
5. 3D transformations

6. Textures and lighting

## Example Quality Standards

- **Self-contained**: Each example runs independently
- **Minimal**: Only code necessary to demonstrate concept

- **Commented**: Explain GPU-specific parts

- **Tested**: Include expected output
- **Progressive**: Build on previous examples

## User Preferences

- Prefers **TypeScript** over JavaScript for WebGPU
- Values **working code** over theory
- Asks questions freely - be patient and thorough
- Appreciates **realistic use cases** over toy examples
- Direct communication, no corporate-speak
- Wants **FAQs documented in source code** as questions arise

## Common Pitfalls to Address

1. **Memory confusion**: Make CPU vs GPU memory crystal clear
2. **Thread indexing math**: Explain `blockIdx.x * blockDim.x + threadIdx.x` thoroughly
3. **Async nature**: GPU launches are asynchronous (important for WebGPU)
4. **Error handling**: CUDA errors are easy to miss without checks

## Success Criteria

User should be able to:
- Understand when GPU is beneficial vs CPU
- Write basic CUDA kernels
- Translate CUDA concepts to WebGPU
- Build simple 3D scenes in WebGPU
- Explain parallel computing to others

---

**Remember**: This user is sharp and learns fast, but GPU programming requires a mental shift from sequential to parallel thinking. Be patient with foundational questions while moving at their pace. **Always document questions in the source file's FAQ section.**
