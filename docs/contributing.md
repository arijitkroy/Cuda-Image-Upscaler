# Contributing notes

## Source map

- Add CLI behavior in `src/main.cu` and document every public option in both `README.md` and `docs/cli.md`.
- Keep CUDA kernel declarations in `include/upscaler.cuh` aligned with definitions in `src/upscaler.cu`.
- Keep the `Pixel` layout as three 8-bit channels unless the PPM I/O and tensor conversion paths are changed together.
- Changes to Real-ESRGAN shapes, normalization, or providers require updates to `src/super_resolution.cu` and `docs/image-pipeline.md`.

## Verification checklist

Before submitting changes:

1. Compile the executable with the documented command.
2. Run `--help` and confirm the help text matches `docs/cli.md`.
3. Run at least nearest, bilinear, and bicubic on `assets/input.ppm`.
4. Exercise invalid CLI values such as a malformed scale and negative sharpen amount.
5. When changing the AI path, verify ONNX Runtime's CUDA provider and run a 4x Real-ESRGAN example.
6. Run `git diff --check` on the intended changes.

## Generated and local files

The repository intentionally tracks its executable, generated sample output, model files, and ONNX Runtime distribution. Do not remove third-party notices or license files when updating those bundles. CUDA compiler temporary intermediates remain ignored.

When a dependency version changes, update the build/runtime document and preserve the dependency's notices and redistribution terms.
