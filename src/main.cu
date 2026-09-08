#include <cuda_runtime.h>
#include "../include/image_types.cuh"
#include "../include/image_io.h"
#include "../include/upscaler.cuh"
#include "../include/cuda_utils.cuh"
#include <cstdio>
#include <cstring>
#include <cstdlib>

const char *upscaleTypeToString(UpscaleType type) {
    switch (type)
    {
    case UpscaleType::NEAREST:
        return "nearest";
    case UpscaleType::BILINEAR:
        return "bilinear";
    default:
        return "unknown";
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: \n%s <input.ppm> [output.ppm] [--SCALE N] [--TYPE nearest|bilinear]\n\n", argv[0]);
        printf("Example: \n%s assets\\input.ppm\n%s assets\\input.ppm --SCALE 4\n%s assets\\input.ppm --TYPE bilinear\n%s assets\\input.ppm output\\upscaled.ppm --SCALE 3\n%s assets\\input.ppm output\\upscaled.ppm --SCALE 3 --TYPE nearest\n", argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }

    const char *inputFilename = argv[1];
    const char *outputFilename = "output/upscaled.ppm";
    UpscaleType upscaleType = UpscaleType::BILINEAR;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--SCALE") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --SCALE requires a value\n");
                return 1;
            }
            SCALE = atoi(argv[i + 1]);
            if (SCALE <= 0) {
                fprintf(stderr, "Error: SCALE must be greater than 0\n");
                return 1;
            }
            i++;
        }
        else if (strcmp(argv[i], "--TYPE") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --TYPE requires a value\n");
                return 1;
            }
            const char *type = argv[i + 1];
            if (strcmp(type, "nearest") == 0) upscaleType = UpscaleType::NEAREST;
            else if (strcmp(type, "bilinear") == 0) upscaleType = UpscaleType::BILINEAR;
            else {
                fprintf(stderr, "Error: Unsupported upscale type: %s\n", type);
                fprintf(stderr, "Error: Supported types: nearest, bilinear\n");
                return 1;
            }
            i++;
        }
        else if (strncmp(argv[i], "--", 2) == 0) {
            fprintf(stderr, "Error: Unknown option: %s\n", argv[i]);
            return 1;
        }
        else {
            outputFilename = argv[i];
        }
    }

    printf("CUDA Image Upscaler\n");
    printf("Input file: %s\n", inputFilename);
    printf("Output file: %s\n", outputFilename);
    printf("Scale Factor: %d\n", SCALE);
    printf("Upscale type: %s\n", upscaleTypeToString(upscaleType));

    Image input = loadPPM(inputFilename);
    if (input.pixels == nullptr) return 1;

    printf("Input size: %d x %d\n", input.width, input.height);

    Image output;
    output.width = input.width * SCALE;
    output.height = input.height * SCALE;

    size_t inputPixelCount = static_cast<size_t>(input.width) * input.height;
    size_t outputPixelCount = static_cast<size_t>(output.width) * output.height;
    size_t inputSize = inputPixelCount * sizeof(Pixel);
    size_t outputSize = outputPixelCount * sizeof(Pixel);

    output.pixels = new Pixel[outputPixelCount];
    Pixel *d_input = nullptr;
    Pixel *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMalloc(&d_output, outputSize));
    CUDA_CHECK(cudaMemcpy(d_input, input.pixels, inputSize, cudaMemcpyHostToDevice));
    
    
    switch (upscaleType) {
        case UpscaleType::NEAREST:
            upscaleNearestGPU(d_input, d_output, input.width, input.height);
            break;
        case UpscaleType::BILINEAR:
            upscaleBilinearGPU(d_input, d_output, input.width, input.height);
            break;
        default:
            fprintf(stderr, "Error: Unsupported upscale type\n");
            CUDA_CHECK(cudaFree(d_input));
            CUDA_CHECK(cudaFree(d_input));
            freeImage(input);
            freeImage(output);
            return 1;
    }


    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.pixels, d_output, outputSize, cudaMemcpyDeviceToHost));

    bool success = savePPM(outputFilename, output);
    if (success) {
        printf("Output saved: %s\n", outputFilename);
        printf("Output size: %d x %d\n", output.width, output.height);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    freeImage(input);
    freeImage(output);

    return success ? 0 : 1;
}