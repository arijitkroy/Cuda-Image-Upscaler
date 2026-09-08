#ifndef IMAGE_TYPES_CUH
#define IMAGE_TYPES_CUH

#include <cstdint>

struct Pixel {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

struct Image {
    int width;
    int height;
    Pixel *pixels;
};

#endif