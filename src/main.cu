#include <cuda_runtime.h>
#include "../include/cuda_utils.cuh"
#include "../include/image_io.h"
#include "../include/image_types.cuh"
#include "../include/super_resolution.cuh"
#include "../include/upscaler.cuh"

#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

int SCALE = 2;

namespace {

struct CommandLineOptions {
    std::string inputFilename;
    std::string outputFilename = "output/upscaled.ppm";
    std::string modelFilename = "model/RealESRGAN_x4plus.onnx";
    int scale = 2;
    float sharpenAmount = 0.0f;
    UpscaleType upscaleType = UpscaleType::BICUBIC;
};

void printUsage(const char* executable) {
    std::cout
        << "Usage: " << executable << " <input.ppm> [output.ppm] [options]\n\n"
        << "Options:\n"
        << "  -s, --scale N          Positive integer scale factor (default: 2)\n"
        << "  -t, --type TYPE        nearest, bilinear, bicubic, or realesrgan (default: bicubic)\n"
        << "      --sharpen AMOUNT   Non-negative sharpening strength (default: 0)\n"
        << "      --model PATH       Real-ESRGAN ONNX model path\n"
        << "  -h, --help             Show this help message\n\n"
        << "Options may appear before or after the optional output path. Both --option value\n"
        << "and --option=value forms are accepted.\n";
}

std::string lowercase(std::string value) {
    for (char& character : value) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    return value;
}

int parsePositiveInt(const std::string& value, const char* option) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno == ERANGE || end == value.c_str() || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
        throw std::runtime_error(std::string(option) + " must be a positive integer; received '" + value + "'.");
    }
    return static_cast<int>(parsed);
}

float parseNonNegativeFloat(const std::string& value, const char* option) {
    char* end = nullptr;
    errno = 0;
    const float parsed = std::strtof(value.c_str(), &end);
    if (errno == ERANGE || end == value.c_str() || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0f) {
        throw std::runtime_error(std::string(option) + " must be a finite, non-negative number; received '" + value + "'.");
    }
    return parsed;
}

UpscaleType parseUpscaleType(const std::string& value) {
    const std::string type = lowercase(value);
    if (type == "nearest") return UpscaleType::NEAREST;
    if (type == "bilinear") return UpscaleType::BILINEAR;
    if (type == "bicubic") return UpscaleType::BICUBIC;
    if (type == "realesrgan") return UpscaleType::REALESRGAN;
    throw std::runtime_error("--type must be nearest, bilinear, bicubic, or realesrgan; received '" + value + "'.");
}

const char* upscaleTypeToString(UpscaleType type) {
    switch (type) {
        case UpscaleType::NEAREST: return "nearest";
        case UpscaleType::BILINEAR: return "bilinear";
        case UpscaleType::BICUBIC: return "bicubic";
        case UpscaleType::REALESRGAN: return "realesrgan";
    }
    return "unknown";
}

std::string optionValue(const std::string& argument, int& index, int argc, char* argv[], const char* option) {
    const std::size_t equals = argument.find('=');
    if (equals != std::string::npos) {
        if (equals == argument.size() - 1) throw std::runtime_error(std::string("Missing value for ") + option + ".");
        return argument.substr(equals + 1);
    }
    if (++index >= argc) throw std::runtime_error(std::string("Missing value for ") + option + ".");
    return argv[index];
}

CommandLineOptions parseCommandLine(int argc, char* argv[]) {
    CommandLineOptions options;
    bool parseOptions = true;
    bool outputProvided = false;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (parseOptions && argument == "--") {
            parseOptions = false;
            continue;
        }
        if (parseOptions && (argument == "-h" || argument == "--help")) {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }

        const std::size_t equals = argument.find('=');
        const std::string option = lowercase(argument.substr(0, equals));
        if (parseOptions && (option == "-s" || option == "--scale")) {
            options.scale = parsePositiveInt(optionValue(argument, index, argc, argv, "--scale"), "--scale");
        } else if (parseOptions && (option == "-t" || option == "--type")) {
            options.upscaleType = parseUpscaleType(optionValue(argument, index, argc, argv, "--type"));
        } else if (parseOptions && option == "--sharpen") {
            options.sharpenAmount = parseNonNegativeFloat(optionValue(argument, index, argc, argv, "--sharpen"), "--sharpen");
        } else if (parseOptions && option == "--model") {
            options.modelFilename = optionValue(argument, index, argc, argv, "--model");
        } else if (parseOptions && !argument.empty() && argument[0] == '-') {
            throw std::runtime_error("Unknown option: '" + argument + "'. Use --help to see supported options.");
        } else if (options.inputFilename.empty()) {
            options.inputFilename = argument;
        } else if (!outputProvided) {
            options.outputFilename = argument;
            outputProvided = true;
        } else {
            throw std::runtime_error("Unexpected positional argument: '" + argument + "'.");
        }
    }

    if (options.inputFilename.empty()) throw std::runtime_error("An input PPM file is required. Use --help for usage.");
    if (options.upscaleType == UpscaleType::REALESRGAN && options.scale != 4) {
        throw std::runtime_error("Real-ESRGAN x4plus requires --scale 4.");
    }
    return options;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const CommandLineOptions options = parseCommandLine(argc, argv);
        SCALE = options.scale;

        std::cout << "Input: " << options.inputFilename << "\n"
                  << "Output: " << options.outputFilename << "\n"
                  << "Scale: " << SCALE << "\n"
                  << "Type: " << upscaleTypeToString(options.upscaleType) << "\n"
                  << "Sharpen: " << options.sharpenAmount << "\n";

        Image input = loadPPM(options.inputFilename.c_str());
        if (!input.pixels) throw std::runtime_error("Failed to load input image.");
        if (input.width > INT_MAX / SCALE || input.height > INT_MAX / SCALE) {
            freeImage(input);
            throw std::runtime_error("Scaled image dimensions exceed supported limits.");
        }

        const int outputWidth = input.width * SCALE;
        const int outputHeight = input.height * SCALE;
        Image output{outputWidth, outputHeight, new Pixel[static_cast<size_t>(outputWidth) * outputHeight]};
        Pixel* dInput = nullptr;
        Pixel* dUpscaled = nullptr;
        Pixel* dOutput = nullptr;

        try {
            const size_t inputBytes = static_cast<size_t>(input.width) * input.height * sizeof(Pixel);
            const size_t outputBytes = static_cast<size_t>(outputWidth) * outputHeight * sizeof(Pixel);
            CUDA_CHECK(cudaMalloc(&dInput, inputBytes));
            CUDA_CHECK(cudaMalloc(&dUpscaled, outputBytes));
            CUDA_CHECK(cudaMalloc(&dOutput, outputBytes));
            CUDA_CHECK(cudaMemcpy(dInput, input.pixels, inputBytes, cudaMemcpyHostToDevice));

            switch (options.upscaleType) {
                case UpscaleType::NEAREST: upscaleNearestGPU(dInput, dUpscaled, input.width, input.height); break;
                case UpscaleType::BILINEAR: upscaleBilinearGPU(dInput, dUpscaled, input.width, input.height); break;
                case UpscaleType::BICUBIC: upscaleBicubicGPU(dInput, dUpscaled, input.width, input.height); break;
                case UpscaleType::REALESRGAN:
                    std::cout << "Loading Real-ESRGAN ONNX model...\n";
                    SuperResolutionEngine(options.modelFilename.c_str()).process(dInput, dUpscaled, input.width, input.height);
                    break;
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            if (options.sharpenAmount > 0.0f) {
                sharpenGPU(dUpscaled, dOutput, outputWidth, outputHeight, options.sharpenAmount);
                CUDA_CHECK(cudaDeviceSynchronize());
            } else {
                CUDA_CHECK(cudaMemcpy(dOutput, dUpscaled, outputBytes, cudaMemcpyDeviceToDevice));
            }
            CUDA_CHECK(cudaMemcpy(output.pixels, dOutput, outputBytes, cudaMemcpyDeviceToHost));
        } catch (...) {
            if (dInput) cudaFree(dInput);
            if (dUpscaled) cudaFree(dUpscaled);
            if (dOutput) cudaFree(dOutput);
            freeImage(input);
            freeImage(output);
            throw;
        }

        CUDA_CHECK(cudaFree(dInput));
        CUDA_CHECK(cudaFree(dUpscaled));
        CUDA_CHECK(cudaFree(dOutput));
        const bool saved = savePPM(options.outputFilename.c_str(), output);
        freeImage(input);
        freeImage(output);
        if (!saved) throw std::runtime_error("Failed to save output image.");

        std::cout << "Upscaling completed successfully.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
