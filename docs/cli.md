# Command-line reference

## Syntax

```text
image_upscaler.exe <input.ppm> [output.ppm] [options]
```

There must be one input path and at most one output path. When omitted, the output path is `output/upscaled.ppm`.

Options may be placed before, between, or after positional paths. Both `--option value` and `--option=value` are valid. Long option names and mode values are case-insensitive, so existing commands using `--SCALE` and `--TYPE` continue to work.

## Options

| Form | Value | Default | Rules |
| --- | --- | --- | --- |
| `-s N`, `--scale N` | integer | `2` | Must be a positive base-10 integer in the supported `int` range. |
| `-t TYPE`, `--type TYPE` | mode | `bicubic` | `nearest`, `bilinear`, `bicubic`, or `realesrgan`. |
| `--sharpen AMOUNT` | float | `0` | Must be finite and non-negative. Fractional values are preserved. |
| `--model PATH` | file path | `model/RealESRGAN_x4plus.onnx` | Used only by `realesrgan`. |
| `-h`, `--help` | none | n/a | Prints usage and exits successfully. |
| `--` | none | n/a | Ends option parsing; remaining arguments are treated as paths. |

## Examples

```powershell
# Default: 2x bicubic, default output path
.\image_upscaler.exe assets\input.ppm

# All options before positional paths
.\image_upscaler.exe --scale=3 --type=BILINEAR assets\input.ppm output\bilinear-3x.ppm

# Short forms, chosen output, and a fractional sharpening strength
.\image_upscaler.exe assets\input.ppm output\nearest.ppm -s 4 -t nearest --sharpen 0.35

# Use a model located outside the default directory
.\image_upscaler.exe assets\input.ppm output\ai.ppm --type realesrgan --scale 4 --model D:\models\realesrgan.onnx
```

## Validation behavior

The program validates all options before allocating CUDA image buffers. The following are errors:

- missing input path;
- unknown option or more than two positional paths;
- a missing option value;
- `--scale 0`, `--scale -1`, `--scale 2x`, or an out-of-range scale;
- negative, non-finite, or malformed sharpening strength;
- an unsupported mode;
- `--type realesrgan` with any scale other than `4`.

Paths beginning with a dash require the option delimiter. For example, `image_upscaler.exe -- -input.ppm output.ppm` treats `-input.ppm` as the input filename.

## Output dimensions

For geometric modes, `outputWidth = inputWidth * scale` and `outputHeight = inputHeight * scale`. The application rejects a scale that would overflow either signed output dimension. Real-ESRGAN always has an x4 output shape.
