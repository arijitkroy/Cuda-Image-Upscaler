#include "../include/image_io.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
using namespace std;

Image loadPPM(const char *filename) {
    Image image;
    image.width = 0;
    image.height = 0;
    image.pixels = nullptr;

    FILE *file = fopen(filename, "rb");

    if (!file) {
        fprintf(stderr, "Failed to open image: %s\n", filename);
        return image;
    }

    char format[3];
    if (fscanf(file, "%2s", format) != 1) {
        fclose(file);
        return image;
    }

    if (format[0] != 'P' || format[1] != '6') {
        fprintf(stderr, "Only P6 PPM images are supported\n");
        return image;
    }

    int max;
    if (fscanf(file, "%d %d", &image.width, &image.height) != 2) {
        fclose(file);
        return image;
    }

    if (fscanf(file, "%d", &max) != 1) {
        fclose(file);
        return image;
    }

    fgetc(file);

    if (max != 255) {
        fprintf(stderr, "Only 8-bit PPM images are supported\n");
        fclose(file);
        return image;
    }

    size_t pixelCount = static_cast<size_t>(image.width) * image.height;
    image.pixels = new Pixel[pixelCount];
    size_t readCount = fread(image.pixels, sizeof(Pixel), pixelCount, file);
    fclose(file);

    if (readCount != pixelCount) {
        fprintf(stderr, "Failed to read image data\n");
        delete[] image.pixels;
        image.width = 0;
        image.height = 0;
    }

    return image;
}

bool savePPM(const char *filename, const Image &image) {
    try {
        filesystem::path outputPath(filename);
        filesystem::path parentDirectory = outputPath.parent_path();

        if (!parentDirectory.empty()) filesystem::create_directories(parentDirectory);
    }
    catch(const filesystem::filesystem_error &error) {
        fprintf(stderr, "Failed to create output directory: %s\n", error.what());
        return false;
    }
    
    FILE *file = fopen(filename, "wb");

    if (!file) {
        fprintf(stderr, "Failed to create output file: %s\n", filename);
        return false;
    }

    fprintf(file, "P6\n%d %d\n255\n", image.width, image.height);

    size_t pixelCount = static_cast<size_t>(image.width) * image.height;
    size_t writeCount = fwrite(image.pixels, sizeof(Pixel), pixelCount, file);
    fclose(file);

    return (writeCount == pixelCount);
}

void freeImage(Image &image) {
    delete[] image.pixels;
    image.width = 0;
    image.height = 0;
    image.pixels = nullptr;
}