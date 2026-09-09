#ifndef SUPER_RESOLUTION_CUH
#define SUPER_RESOLUTION_CUH

#include "image_types.cuh"

class SuperResolutionEngine {
public:
    SuperResolutionEngine(const char *modelPath);
    ~SuperResolutionEngine();
    SuperResolutionEngine(const SuperResolutionEngine&) = delete;
    SuperResolutionEngine& operator=(const SuperResolutionEngine&) = delete;
    void process(const Pixel *d_input, Pixel *d_output, int inputWidth, int inputHeight);
private:
    struct Impl;
    Impl* impl;
};

#endif