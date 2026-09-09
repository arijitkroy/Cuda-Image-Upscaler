#include <cuda_runtime.h>
#include "../include/upscaler.cuh"
#include "../include/cuda_utils.cuh"

__global__ void upscaleNearestKernel(const Pixel *input, Pixel *output, int inputWidth, int inputHeight, int scale) {
    int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    int outputY = blockIdx.y * blockDim.y + threadIdx.y;
    int outputWidth = inputWidth * scale;
    int outputHeight = inputHeight * scale;
    if (outputX >= outputWidth || outputY >= outputHeight) return;

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
    if (outputX >= outputWidth || outputY >= outputHeight) return;

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

__device__ float cubicWeight(float x) {
    x = fabs(x);
    if (x <= 1.0f) return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
    if (x < 2.0f) return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
    return 0.0f;
}

__device__ float sampleChannel(const Pixel *input, int width, int height, float x, float y, int channel) {
    int baseX = static_cast<int>(floor(x));
    int baseY = static_cast<int>(floor(y));
    float result = 0.0f;
    float totalWeight = 0.0f;

    for (int j = -1; j <= 2; j++) {
        for (int i = -1; i <= 2; i++) {
            int sampleX = baseX + i;
            int sampleY = baseY + j;
            sampleX = max(0, min(sampleX, width - 1));
            sampleY = max(0, min(sampleY, height - 1));
            float weightX = cubicWeight(x - sampleX);
            float weightY = cubicWeight(y - sampleY);
            float weight = weightX * weightY;
            Pixel pixel = input[sampleY * width + sampleX];
            float value;
            if (channel == 0) value = pixel.r;
            else if (channel == 1) value = pixel.g;
            else value = pixel.b;
            result += value * weight;
            totalWeight += weight;
        }
    }
    if (totalWeight > 0.0f) result /= totalWeight;
    return fminf(255.0f, fmaxf(0.0f, result));
}

__global__ void upscaleBicubicKernel(const Pixel *input, Pixel *output, int inputWidth, int inputHeight, int scale) {
    int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    int outputY = blockIdx.y * blockDim.y + threadIdx.y;
    int outputWidth = inputWidth * scale;
    int outputHeight = inputHeight * scale;
    if (outputX >= outputWidth || outputY >= outputHeight) return;

    float inputX = static_cast<float>(outputX) / scale;
    float inputY = static_cast<float>(outputY) / scale;

    Pixel result;
    result.r = static_cast<unsigned char>(sampleChannel(input, inputWidth, inputHeight, inputX, inputY, 0));
    result.g = static_cast<unsigned char>(sampleChannel(input, inputWidth, inputHeight, inputX, inputY, 1));
    result.b = static_cast<unsigned char>(sampleChannel(input, inputWidth, inputHeight, inputX, inputY, 2));

    output[outputY * outputWidth + outputX] = result;
}

void upscaleBicubicGPU(const Pixel *d_input, Pixel *d_output, int inputWidth, int inputHeight) {
    int outputWidth = inputWidth * SCALE;
    int outputHeight = inputHeight * SCALE;
    dim3 threads(16, 16);
    dim3 blocks(
        (outputWidth + threads.x - 1) / threads.x,
        (outputHeight + threads.y - 1) / threads.y
    );
    upscaleBicubicKernel<<<blocks, threads>>>(d_input, d_output, inputWidth, inputHeight, SCALE);
    CUDA_CHECK(cudaGetLastError());
}

__device__ int clampChannel(int value) {
    return max(0, min(255, value));
}

__global__ void sharpenKernel(const Pixel *input, Pixel *output, int width, int height, float amount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int centerIndex = y * width + x;
    Pixel center = input[centerIndex];
    Pixel left = input[y * width + max(0, x - 1)];
    Pixel right = input[y * width + min(width - 1, x + 1)];
    Pixel top = input[max(0, y - 1) * width + x];
    Pixel bottom = input[min(height - 1, y + 1) * width + x];

    float r = center.r + amount * (4.0f * center.r - left.r - right.r - top.r - bottom.r);
    float g = center.g + amount * (4.0f * center.g - left.g - right.g - top.g - bottom.g);
    float b = center.b + amount * (4.0f * center.b - left.b - right.b - top.b - bottom.b);

    output[centerIndex].r = static_cast<unsigned char>(clampChannel(static_cast<int>(r)));
    output[centerIndex].g = static_cast<unsigned char>(clampChannel(static_cast<int>(g)));
    output[centerIndex].b = static_cast<unsigned char>(clampChannel(static_cast<int>(b)));
}

void sharpenGPU(const Pixel *d_input, Pixel *d_output, int width, int height, float amount) {
    dim3 threads(16, 16);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y
    );
    sharpenKernel<<<blocks, threads>>>(d_input, d_output, width, height, amount);
    CUDA_CHECK(cudaGetLastError());
}
