#include <cuda_runtime.h>
#include "../include/upscaler.cuh"
#include "../include/cuda_utils.cuh"

int SCALE = 2;

__global__ void upscaleNearestKernel(const Pixel *input, Pixel *output, int inputWidth, int inputHeight, int scale) {
    int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    int outputY = blockIdx.y * blockDim.y + threadIdx.y;
    int outputWidth = inputWidth * scale;
    int outputHeight = inputHeight * scale;
    if (outputX >= outputWidth | outputY >= outputHeight) return;

    int inputX = outputX / scale;
    int inputY = outputY / scale;
    int inputIndex = inputY * inputWidth + inputX;
    int outputIndex = outputY * outputWidth + outputX;

    output[outputIndex] = input[inputIndex];
}

void upscaleNearestGPU(const Pixel *d_input, Pixel *d_output, int inputWidth, int inputHeight) {
    int outputWidth = inputWidth * SCALE;
    int outputHeight = inputHeight * SCALE;
    dim3 threads(16, 16);
    dim3 blocks(
        (outputWidth + threads.x - 1) / threads.x,
        (outputHeight + threads.y - 1) / threads.y
    );
    upscaleNearestKernel<<<blocks, threads>>>(d_input, d_output, inputWidth, inputHeight, SCALE);
    CUDA_CHECK(cudaGetLastError());
}

__device__ unsigned char interpolate(unsigned char a, unsigned char b, float factor) {
    float value = a + factor * (b - a);
    value = fminf(255.0f, fmaxf(0.0f, value));
    return static_cast<unsigned char>(value);
}

__global__ void upscaleBilinearKernel(const Pixel *input, Pixel *output, int inputWidth, int inputHeight, int scale) {
    int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    int outputY = blockIdx.y * blockDim.y + threadIdx.y;
    int outputWidth = inputWidth * scale;
    int outputHeight = inputHeight * scale;
    if (outputX >= outputWidth | outputY >= outputHeight) return;

    float inputX = static_cast<float>(outputX) / scale;
    float inputY = static_cast<float>(outputY) / scale;
    int x0 = static_cast<int>(floorf(inputX));
    int y0 = static_cast<int>(floorf(inputY));
    int x1 = min(x0 + 1, inputWidth - 1);
    int y1 = min(y0 + 1, inputHeight - 1);
    float xFraction = inputX - x0;
    float yFraction = inputY - y0;

    Pixel topLeft = input[y0 * inputWidth + x0];
    Pixel topRight = input[y0 * inputWidth + x1];
    Pixel bottomLeft = input[y1 * inputWidth + x0];
    Pixel bottomRight = input[y1 * inputWidth + x1];
    Pixel top;
    Pixel bottom;
    Pixel result;

    top.r = interpolate(topLeft.r, topRight.r, xFraction);
    top.g = interpolate(topLeft.g, topRight.g, xFraction);
    top.b = interpolate(topLeft.b, topRight.b, xFraction);

    bottom.r = interpolate(bottomLeft.r, bottomRight.r, xFraction);
    bottom.g = interpolate(bottomLeft.g, bottomRight.g, xFraction);
    bottom.b = interpolate(bottomLeft.b, bottomRight.b, xFraction);

    result.r = interpolate(top.r, bottom.r, yFraction);
    result.g = interpolate(top.g, bottom.g, yFraction);
    result.b = interpolate(top.b, bottom.b, yFraction);

    int outputIndex = outputY * outputWidth + outputX;
    output[outputIndex] = result;
}

void upscaleBilinearGPU(const Pixel *d_input, Pixel *d_output, int inputWidth, int inputHeight) {
    int outputWidth = inputWidth * SCALE;
    int outputHeight = inputHeight * SCALE;
    dim3 threads(16, 16);
    dim3 blocks(
        (outputWidth + threads.x - 1) / threads.x,
        (outputHeight + threads.y - 1) / threads.y
    );
    upscaleBilinearKernel<<<blocks, threads>>>(d_input, d_output, inputWidth, inputHeight, SCALE);
    CUDA_CHECK(cudaGetLastError());
}