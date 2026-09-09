#ifndef UPSCALER_CUH
#define UPSCALER_CUH

#include <image_types.cuh>

extern int SCALE;

enum class UpscaleType {
    NEAREST, 
    BILINEAR,
    BICUBIC,
    REALESRGAN
};

void upscaleNearestGPU(const Pixel *d_input, Pixel *d_output, int inputWidth, int inputHeight);
void upscaleBilinearGPU(const Pixel *d_input, Pixel *d_output, int inputWidth, int inputHeight);
void upscaleBicubicGPU(const Pixel *d_input, Pixel *d_output, int inputWidth, int inputHeight);
void sharpenGPU(const Pixel *d_input, Pixel *d_output, int width, int height, float amount);

#endif
