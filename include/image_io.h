#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <image_types.cuh>

Image loadPPM(const char *filename);
bool savePPM(const char *filename, const Image &image);
void freeImage(Image &image);

#endif