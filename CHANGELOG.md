# Changelog

All notable changes to this project are documented in this file.

## Unreleased

### Changed

- Reworked command-line parsing to support an optional output path, flags before or after positional paths, short options, `--option=value`, and the `--` delimiter.
- Made option names and interpolation-type values case-insensitive while retaining compatibility with `--SCALE` and `--TYPE`.
- Documented the current bicubic, sharpening, and Real-ESRGAN capabilities, build requirements, and runtime dependencies.

### Fixed

- Reject malformed, out-of-range, zero, or negative scale factors without uncaught conversion exceptions.
- Reject negative, non-finite, and malformed sharpening strengths.
- Preserve fractional sharpening strengths through the host and CUDA interfaces.
- Use logical bounds checks in CUDA kernels.
