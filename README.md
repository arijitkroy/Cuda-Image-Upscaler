# CUDA Image Upscaler

CUDA Image Upscaler is a Windows-oriented C++17 command-line utility for enlarging binary P6 PPM images on an NVIDIA GPU. It offers nearest-neighbor, bilinear, and bicubic resampling, an optional sharpening pass, and optional CUDA-backed Real-ESRGAN x4 inference.

## Features

- CUDA implementations of nearest-neighbor, bilinear, and bicubic interpolation
- Optional non-negative sharpening strength, including fractional values
- Real-ESRGAN x4 inference through ONNX Runtime's CUDA execution provider
- Strict P6 PPM (8-bit RGB) reader and writer; output folders are created automatically
- Consistent, order-independent command-line options with useful validation errors
- Pillow utilities for converting to and from common image formats

## Project layout

```text
assets/                    Sample images
include/                   CUDA and image-processing interfaces
model/                     Real-ESRGAN ONNX model (local, not versioned)
src/                       Application, CUDA kernels, and ONNX integration
third_party/onnxruntime/   Local ONNX Runtime SDK (local, not versioned)
tools/                     Pillow conversion and ONNX model helper scripts
output/                    Generated images (local)
```

## Requirements

- Windows, an NVIDIA GPU, and a CUDA-capable driver
- CUDA Toolkit with `nvcc` and a C++17-capable supported host compiler
- Python 3.8+ and Pillow only for the conversion scripts
- For `realesrgan`: the ONNX Runtime GPU SDK and `model/RealESRGAN_x4plus.onnx`

The geometric interpolation modes do not require ONNX Runtime at execution time. The current application source includes the optional Real-ESRGAN backend, so the ONNX Runtime development headers and import library are required when compiling this checkout.

## Build

Place an ONNX Runtime GPU SDK under `third_party/onnxruntime` (or adjust the paths below). Its `include` directory and `lib/onnxruntime.lib` are used at link time. Build from the repository root:

```powershell
nvcc -std=c++17 -Iinclude -Ithird_party\onnxruntime\include src\main.cu src\upscaler.cu src\super_resolution.cu src\image_io.cpp -Lthird_party\onnxruntime\lib -lonnxruntime -o image_upscaler.exe
```

For Real-ESRGAN runtime support, copy the matching ONNX Runtime DLLs beside `image_upscaler.exe` (including the CUDA provider DLLs) and place the ONNX model at `model\RealESRGAN_x4plus.onnx`. CUDA Toolkit and ONNX Runtime versions must be compatible.

## Usage

```text
image_upscaler.exe <input.ppm> [output.ppm] [options]
```

`input.ppm` is required. `output.ppm` is optional and defaults to `output/upscaled.ppm`. Options can appear before or after the output path. Use either a space or `=` between a long option and its value; option names and interpolation types are case-insensitive.

| Option | Default | Description |
| --- | --- | --- |
| `-s N`, `--scale N` | `2` | Positive integer multiplier for each image dimension. |
| `-t TYPE`, `--type TYPE` | `bicubic` | `nearest`, `bilinear`, `bicubic`, or `realesrgan`. |
| `--sharpen AMOUNT` | `0` | Finite, non-negative post-processing strength. |
| `--model PATH` | `model/RealESRGAN_x4plus.onnx` | ONNX model path used by `realesrgan`. |
| `-h`, `--help` | — | Display usage. |

Use `--` before a positional path beginning with a dash.

```powershell
# Default: 2x bicubic output at output\upscaled.ppm
.\image_upscaler.exe assets\input.ppm

# Options may precede the output path; uppercase legacy spelling still works
.\image_upscaler.exe --SCALE=3 --TYPE BILINEAR assets\input.ppm output\bilinear-3x.ppm

# Nearest-neighbor with a fractional sharpening pass
.\image_upscaler.exe assets\input.ppm output\nearest.ppm -s 4 -t nearest --sharpen 0.35

# Real-ESRGAN x4 is intentionally restricted to scale 4
.\image_upscaler.exe assets\input.ppm output\realesrgan.ppm --type=realesrgan --scale=4
```

Invalid numeric values such as `--scale 2x`, `--scale 0`, `--sharpen -1`, and non-finite sharpening values are rejected before CUDA work begins.

## Input and output format

The executable accepts and writes binary P6 PPM files with three RGB channels and `maxval` of `255`. Convert other image formats with Pillow:

```powershell
python -m pip install -r tools\requirements.txt
python tools\convert_to_ppm.py assets\photo.png assets\photo.ppm
.\image_upscaler.exe assets\photo.ppm output\photo-4x.ppm --scale 4 --type bicubic
python tools\convert_from_ppm.py output\photo-4x.ppm output\photo-4x.png
```

## Modes

- **nearest** preserves source pixels in square blocks and is useful for pixel art.
- **bilinear** blends four neighbouring source samples for smooth enlargement.
- **bicubic** samples a 4×4 neighbourhood for a sharper, smoother geometric result.
- **realesrgan** runs the RealESRGAN x4plus ONNX model on CUDA. It requires `--scale 4`; it is not a general arbitrary-scale mode.

All modes use 16×16 CUDA thread blocks. Memory use grows approximately with the square of the scale factor.

## Utilities

- `tools/convert_to_ppm.py`: Converts any Pillow-supported input to an RGB PPM file.
- `tools/convert_from_ppm.py`: Converts a PPM result to PNG.
- `tools/export_onnx.py`: Exports `RealESRGAN_x4plus.pth` to ONNX; it requires PyTorch and BasicSR.
- `tools/test_onnx.py` and `src/onnx_test.cpp`: Report ONNX Runtime CUDA-provider availability.

## Limitations

- P6 PPM is the only native image format.
- The application uses CUDA device 0.
- Real-ESRGAN model and runtime binaries are local dependencies and are intentionally excluded from Git history.
- No license file is currently included; add one before distributing the project.

See [CHANGELOG.md](CHANGELOG.md) for release notes.
