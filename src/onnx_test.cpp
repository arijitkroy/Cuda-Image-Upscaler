#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CudaImageUpscaler");
    Ort::SessionOptions options;
    auto providers = Ort::GetAvailableProviders();

    std::cout << "ONNX Runtime version: " << Ort::GetVersionString() << "\n";
    std::cout << "Execution providers:\n";
    for (const auto& provider : providers) std::cout << "  " << provider << "\n";

    bool cudaAvailable = false;
    for (const auto& provider : providers) {
        if (provider == "CUDAExecutionProvider") cudaAvailable = true;
    }

    if (!cudaAvailable) {
        std::cerr << "CUDAExecutionProvider is NOT available.\n";
        return 1;
    }

    std::cout << "CUDAExecutionProvider is available.\n";
    return 0;
}