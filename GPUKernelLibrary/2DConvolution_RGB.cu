#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "2DConvolution_RGB.cuh"
#include <iostream>
using namespace std;

__global__ void convolution_RGB(float* inImage, float* opImage, float* filter,
    int paddedWidth, int paddedHeight, int width, int height, int channels, int filterDim) {

    int index = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;

    int filterRadius = filterDim / 2;

    for (int i = index; i < channels * width * height; i += stride) {
        int z = i % channels;
        int x = (i / channels) % width;
        int y = i / (channels * width);

        float sum = 0.0f;

        int paddedX = x + filterRadius;
        int paddedY = y + filterRadius;

        for (int j = -filterRadius; j <= filterRadius; j++) {
            for (int k = -filterRadius; k <= filterRadius; k++) {
                int x1 = paddedX + j;
                int y1 = paddedY + k;

                if (x1 >= 0 && x1 < paddedWidth && y1 >= 0 && y1 < paddedHeight) {
                    sum += filter[(j + filterRadius) * filterDim + (k + filterRadius)] * inImage[(y1 * paddedWidth + x1) * channels + z];
                }
            }
        }

        opImage[(y * width + x) * channels + z] = sum;
    }
}

