# CUDA Image Upscaler

A compact CUDA C++ command-line application that enlarges 8-bit RGB PPM images on the GPU. It provides two interpolation modes—nearest-neighbor and bilinear—and includes Python utilities for converting common image formats to and from the PPM format used by the CUDA pipeline.

## Features

- GPU-accelerated image upscaling using CUDA kernels
- Nearest-neighbor interpolation for crisp, pixel-preserving enlargement
- Bilinear interpolation for smoother enlarged images
- Configurable positive integer scale factor
- Binary P6 PPM input and output with 8-bit RGB channels
- Automatic creation of output directories
- Pillow-based conversion helpers for using PNG, JPEG, and other Pillow-supported formats
- CUDA error checking after memory operations and kernel launches

## Project layout

```text
.
├── assets/
│   └── input.ppm              Sample P6 PPM input image
├── include/
│   ├── cuda_utils.cuh         CUDA error-checking macro
│   ├── image_io.h             PPM load/save declarations
│   ├── image_types.cuh        Pixel and Image data structures
│   └── upscaler.cuh           GPU upscaler interface
├── output/                    Default location for generated images
├── src/
│   ├── image_io.cpp           P6 PPM reader and writer
│   ├── main.cu                Command-line parsing and CUDA workflow
│   └── upscaler.cu            Nearest-neighbor and bilinear kernels
├── tools/
│   ├── convert_from_ppm.py    Converts a PPM result to PNG
│   ├── convert_to_ppm.py      Converts an image to PPM
│   └── requirements.txt       Python dependencies
└── image_upscaler.exe         Windows build artifact, if built locally
```

## Requirements

- An NVIDIA GPU with a CUDA-capable driver
- NVIDIA CUDA Toolkit with `nvcc`
- A C++17-capable host compiler supported by the installed CUDA Toolkit
- Python 3.8 or newer and Pillow, only when using the conversion helpers

The native application has no third-party C++ library dependencies.

## Build

From the repository root, compile all CUDA and C++ sources with C++17 enabled:

```powershell
nvcc -std=c++17 -Iinclude src/main.cu src/upscaler.cu src/image_io.cpp -o image_upscaler.exe
```

On Unix-like systems, choose an appropriate output name:

```bash
nvcc -std=c++17 -Iinclude src/main.cu src/upscaler.cu src/image_io.cpp -o image_upscaler
```

## Usage

```text
image_upscaler <input.ppm> [output.ppm] [--SCALE N] [--TYPE nearest|bilinear]
```

The input image is required. If no output path is supplied, the program writes to `output/upscaled.ppm`. The default interpolation mode is `bilinear`, and the default scale factor is `2`.

Examples:

```powershell
# Use defaults: 2x bilinear output at output/upscaled.ppm
.\image_upscaler.exe assets\input.ppm

# Create a 4x bilinear result
.\image_upscaler.exe assets\input.ppm --SCALE 4

# Create a 3x nearest-neighbor result at a chosen path
.\image_upscaler.exe assets\input.ppm output\nearest-3x.ppm --SCALE 3 --TYPE nearest
```

The program reports the selected options and input/output dimensions. For example, a `500 x 375` source scaled by `2` produces a `1000 x 750` image.

### Command-line options

| Option | Values | Default | Description |
| --- | --- | --- | --- |
| Positional `input.ppm` | Path to a PPM file | Required | Source image to upscale. |
| Positional `output.ppm` | Path | `output/upscaled.ppm` | Destination file. Parent directories are created when needed. |
| `--SCALE N` | Integer greater than zero | `2` | Multiplier applied to both image dimensions. |
| `--TYPE` | `nearest`, `bilinear` | `bilinear` | Interpolation algorithm. |

## Input and output format

The CUDA executable accepts only binary PPM files with the following characteristics:

- Magic number: `P6`
- Three RGB channels
- 8 bits per channel (`maxval` must be `255`)

The `Pixel` structure contains exactly three unsigned-byte channels: red, green, and blue. The output is also written as a P6 PPM image.

## Interpolation modes

### Nearest neighbor

Each output pixel is assigned the value of the corresponding source pixel using integer coordinate division. This keeps original pixel values intact and is useful for pixel art or when sharp block boundaries are preferred.

### Bilinear

For each output coordinate, the kernel maps the coordinate to source space, samples its four surrounding source pixels, and interpolates each RGB channel horizontally and then vertically. At image edges, sample coordinates are clamped to the last valid row or column. This typically produces a smoother visual result than nearest-neighbor scaling.

Both kernels use a two-dimensional `16 x 16` CUDA thread block. The grid dimensions are rounded up so partial edge blocks remain safe; threads outside the output bounds return without writing.

## Converting other image formats

Install the Python dependency:

```powershell
python -m pip install -r tools\requirements.txt
```

Convert an image to a CUDA-compatible PPM file:

```powershell
python tools\convert_to_ppm.py assets\camel.jpg assets\camel.ppm
```

Run the upscaler, then convert the PPM result to PNG:

```powershell
.\image_upscaler.exe assets\camel.ppm output\camel-4x.ppm --SCALE 4 --TYPE bilinear
python tools\convert_from_ppm.py output\camel-4x.ppm output\camel-4x.png
```

The conversion scripts open the source through Pillow and convert it to RGB before saving. `convert_from_ppm.py` writes PNG output.

## Processing flow

```text
P6 PPM input
    |
    v
Host memory allocation and PPM decoding
    |
    v
Copy RGB pixels to CUDA device memory
    |
    v
Nearest-neighbor or bilinear CUDA kernel
    |
    v
Copy enlarged pixels back to host memory
    |
    v
P6 PPM output
```

## Error handling and limitations

- CUDA allocation, memory-copy, launch, and synchronization failures are checked and reported.
- Invalid interpolation types, unknown options, missing option values, and non-positive scale factors are rejected.
- The executable currently supports P6 PPM input only; use the Python converter for JPEG, PNG, and other formats.
- This is a geometric resampler, not an AI super-resolution model. It does not infer new image detail.
- Very large scale factors increase output width and height linearly, but pixel count and GPU memory use grow with the square of the scale factor.

## Verification

The project was verified with CUDA Toolkit 13.3 by running:

```powershell
.\image_upscaler.exe assets\input.ppm --SCALE 2 --TYPE bilinear
```

The bundled `assets/input.ppm` is `500 x 375`; the command produced `output/upscaled.ppm` at `1000 x 750`.

## License

No license file is currently included. Add a license before distributing or reusing this project under explicit terms.
