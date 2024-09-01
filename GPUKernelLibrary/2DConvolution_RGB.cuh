#ifndef CONV_RGB
#define CONV_RGB

#include "cuda_runtime.h"

__global__ void convolution_RGB(float* inImage, float* opImage, float* filter,
    int paddedWidth, int paddedHeight, int width, int height, int channels, int filterDim);

#endif
