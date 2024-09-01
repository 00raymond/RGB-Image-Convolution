#ifndef IMAGE_PADDING_CUH
#define IMAGE_PADDING_CUH

#include "cuda_runtime.h"

__global__ 
void imagePaddingGPU(float* paddedImageArray, float* imageArray, int rows, int cols, int channels, int paddingSize);

#endif
