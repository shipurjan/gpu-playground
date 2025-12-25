# WebGPU Examples

Browser-based GPU programming examples using WebGPU and TypeScript.

## Quick Start

```bash
npm install
npm run dev
```

This will auto-compile TypeScript on changes and start a local server at `http://localhost:8000`.

## Available Scripts

- `npm run dev` - Run TypeScript compiler (watch mode) + HTTP server
- `npm run build` - Compile TypeScript once
- `npm run watch` - Compile TypeScript in watch mode
- `npm run serve` - Start HTTP server on port 8000

## Browser Support

WebGPU requires:
- **Chrome/Edge**: Version 113 or higher
- **Firefox**: Enable `dom.webgpu.enabled` in `about:config`
- **Safari**: Technology Preview (experimental)

Check support at: https://caniuse.com/webgpu

## Examples

- `01-hello-gpu.ts` - Basic GPU compute (equivalent to CUDA hello world)
