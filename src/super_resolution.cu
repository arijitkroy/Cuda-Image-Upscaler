#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <string>
#include <memory>
#include <windows.h>
#include "../include/super_resolution.cuh"
#include "../include/cuda_utils.cuh"

__global__ void pixelsToTensorKernel(const Pixel* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int p = y * width + x;
    int plane = width * height;
    output[p] = static_cast<float>(input[p].r) / 255.0f;
    output[plane + p] = static_cast<float>(input[p].g) / 255.0f;
    output[2 * plane + p] = static_cast<float>(input[p].b) / 255.0f;
}

__global__ void tensorToPixelsKernel(const float* input, Pixel* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int p = y * width + x;
    int plane = width * height;
    float r = fminf(fmaxf(input[p], 0.0f), 1.0f);
    float g = fminf(fmaxf(input[plane + p], 0.0f), 1.0f);
    float b = fminf(fmaxf(input[2 * plane + p], 0.0f), 1.0f);
    output[p].r = static_cast<unsigned char>(r * 255.0f + 0.5f);
    output[p].g = static_cast<unsigned char>(g * 255.0f + 0.5f);
    output[p].b = static_cast<unsigned char>(b * 255.0f + 0.5f);
}

struct SuperResolutionEngine::Impl {
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;

    Impl(const char* modelPath)
        : env(ORT_LOGGING_LEVEL_WARNING, "CudaImageUpscaler"), sessionOptions(), session(nullptr) {
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        OrtCUDAProviderOptions cudaOptions{};
        cudaOptions.device_id = 0;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);

        int length = MultiByteToWideChar(CP_UTF8, 0, modelPath, -1, nullptr, 0);
        if (length <= 0) throw std::runtime_error("Failed to convert model path.");

        std::wstring widePath(length, L'\0');
        if (MultiByteToWideChar(CP_UTF8, 0, modelPath, -1, widePath.data(), length) <= 0)
            throw std::runtime_error("Failed to convert model path.");

        session = std::make_unique<Ort::Session>(env, widePath.c_str(), sessionOptions);
    }
};

SuperResolutionEngine::SuperResolutionEngine(const char* modelPath) {
    try {
        impl = new Impl(modelPath);
    } catch (const Ort::Exception& error) {
        throw std::runtime_error(std::string("Failed to initialize Real-ESRGAN: ") + error.what());
    }
}

SuperResolutionEngine::~SuperResolutionEngine() {
    delete impl;
}

void SuperResolutionEngine::process(const Pixel* d_input, Pixel* d_output, int inputWidth, int inputHeight) {
    if (!d_input || !d_output || inputWidth <= 0 || inputHeight <= 0)
        throw std::runtime_error("Invalid super-resolution input.");

    const int outputWidth = inputWidth * 4;
    const int outputHeight = inputHeight * 4;
    const size_t inputPixels = static_cast<size_t>(inputWidth) * inputHeight;
    const size_t outputPixels = static_cast<size_t>(outputWidth) * outputHeight;
    const size_t inputElements = inputPixels * 3;
    const size_t outputElements = outputPixels * 3;

    float* d_inputTensor = nullptr;
    float* d_outputTensor = nullptr;

    try {
        CUDA_CHECK(cudaMalloc(&d_inputTensor, inputElements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outputTensor, outputElements * sizeof(float)));

        dim3 threads(16, 16);
        dim3 inputBlocks((inputWidth + 15) / 16, (inputHeight + 15) / 16);

        pixelsToTensorKernel<<<inputBlocks, threads>>>(
            d_input,
            d_inputTensor,
            inputWidth,
            inputHeight
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int64_t inputShape[] = {1, 3, inputHeight, inputWidth};
        int64_t outputShape[] = {1, 3, outputHeight, outputWidth};

        Ort::MemoryInfo cudaMemory(
            "Cuda",
            OrtAllocatorType::OrtDeviceAllocator,
            0,
            OrtMemTypeDefault
        );

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            cudaMemory,
            d_inputTensor,
            inputElements,
            inputShape,
            4
        );

        Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
            cudaMemory,
            d_outputTensor,
            outputElements,
            outputShape,
            4
        );

        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = impl->session->GetInputNameAllocated(0, allocator);
        auto outputName = impl->session->GetOutputNameAllocated(0, allocator);

        const char* inputNames[] = {inputName.get()};
        const char* outputNames[] = {outputName.get()};

        impl->session->Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            &outputTensor,
            1
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        dim3 outputBlocks((outputWidth + 15) / 16, (outputHeight + 15) / 16);

        tensorToPixelsKernel<<<outputBlocks, threads>>>(
            d_outputTensor,
            d_output,
            outputWidth,
            outputHeight
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_inputTensor));
        CUDA_CHECK(cudaFree(d_outputTensor));
    }
    catch (...) {
        if (d_inputTensor) cudaFree(d_inputTensor);
        if (d_outputTensor) cudaFree(d_outputTensor);
        throw;
    }
}