# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`img-renew` is a Rust image processing tool for upscaling and sharpening images. It supports two methods:
1. **Traditional processing**: Uses convolution kernels with Lanczos3 interpolation
2. **AI-based enhancement**: Uses ONNX models (Real-ESRGAN) for super-resolution

## Build and Run Commands

### Build the project
```bash
cargo build
cargo build --release
```

### Traditional mode (2x upscale + sharpening)
```bash
cargo run -- <input_image> <output_image> <intensity>
# Example:
cargo run -- test.png renewed.png 1.5
```

Intensity values:
- `1.5`: Standard sharpening (balanced)
- `2.0-3.0`: Strong sharpening (more aggressive)
- `0.5-1.0`: Mild sharpening (subtle)

### AI mode (ONNX model enhancement)
```bash
cargo run -- --onnx <model_path> <input_image> <output_image>
# Example:
cargo run -- --onnx ./model/Real-ESRGAN-x4plus_float/model.onnx test.png output.png
```

Use `test-onnx.sh` for quick AI testing with Real-ESRGAN model.

### Testing
```bash
# Traditional mode test
./test.sh

# AI mode test
./test-onnx.sh
```

## Architecture

### Core Structure
- **`ImageProcessor`**: Main struct handling all image operations
  - `new()`: Load image from file
  - `resize()`: 2x upscaling using Lanczos3 filter
  - `sharpen()`: Traditional convolution-based sharpening
  - `sharpen_ai()`: AI model inference using tract-onnx
  - `save()`: Export processed image

### Processing Flow

#### Traditional Method
1. Load image → 2. Resize (2x) → 3. Apply sharpening kernel → 4. Save

#### AI Method
1. Load image → 2. Preprocess (resize to 128x128, normalize to [0,1], convert to NCHW tensor) → 3. Run ONNX inference → 4. Postprocess (denormalize, convert back to RGB) → 5. Resize to 2x original size → 6. Save

### Sharpening Kernel Mathematics

The `create_sharpen_kernel(intensity)` function generates a 3x3 convolution matrix:
```
[neighbor_weight, neighbor_weight, neighbor_weight]
[neighbor_weight, center_weight,  neighbor_weight]
[neighbor_weight, neighbor_weight, neighbor_weight]
```

Where:
- `neighbor_weight = -intensity`
- `center_weight = 1.0 - (8.0 * neighbor_weight)`

**Key constraint**: All kernel values must sum to 1.0 to preserve brightness.

Example for intensity=1.5:
- neighbor_weight = -1.5
- center_weight = 1.0 - (8.0 × -1.5) = 1.0 + 12.0 = 13.0
- Kernel: `[[-1.5, -1.5, -1.5], [-1.5, 13.0, -1.5], [-1.5, -1.5, -1.5]]`

## Dependencies

Key crates:
- `image`: Core image I/O and operations
- `imageproc`: Advanced processing (convolution)
- `tract-onnx`: ONNX model inference (CPU-only, pure Rust implementation)
- `ndarray`: Tensor operations for AI model I/O

## AI Model Requirements

- Format: ONNX (.onnx files)
- Input: RGB image tensor [1, 3, H, W], normalized to [0, 1]
- Output: RGB image tensor [1, 3, H', W'], normalized to [0, 1]
- Recommended models for text clarity:
  - **Real-ESRGAN-x4plus**: General-purpose super-resolution (HuggingFace: qualcomm/Real-ESRGAN-x4plus)
  - **TextSR**: Specialized for scene text images with OCR guidance
  - **GOT-OCR2.0**: Combined OCR and high-resolution processing for text
- Place models in `./model/` directory

## GPU Support

**Current Status**: CPU-only (using `tract-onnx`)

**Why no GPU support?**
- `tract-onnx` is a pure Rust implementation without GPU acceleration
- Benefit: No external dependencies, lightweight, works everywhere
- Trade-off: Slower inference compared to GPU

**For GPU acceleration**, consider these alternatives:
1. **ort crate** (Rust): Supports CUDA/DirectML/TensorRT, but requires:
   - Cargo features: `cuda`, `directml`, `download-binaries`
   - Compatible MSVC toolchain
   - ONNX Runtime prebuilt binaries
2. **Python + onnxruntime-gpu**: Most stable option for GPU inference
   - Use Python for AI processing, Rust for traditional image ops
   - Call Python script from Rust using `std::process::Command`
3. **Manual ONNX Runtime build**: Compile from source with GPU support

## Notes

- AI mode currently runs on CPU (tract-onnx)
- Traditional mode applies edge-preserving processing (skips image borders)
- The codebase is documented in Traditional Chinese (繁體中文)
- For production use with large images, consider GPU-accelerated alternatives