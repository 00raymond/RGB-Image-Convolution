#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "2DConvolution_RGB.cuh"

__global__ void convolution_RGB(float* inImage, float* opImage, float* filter, int width, int height, int channels, int filterDim) {
    int index = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;

    int filterRadius = filterDim / 2;

    for (int i = index; i < width * height * channels; i += stride) {
        int z = i % channels;
        int x = (i / channels) % width;
        int y = (i / channels) / width;

        float sum = 0.0f;

        // Apply the filter to each pixel
        for (int j = -filterRadius; j <= filterRadius; j++) {
            for (int k = -filterRadius; k <= filterRadius; k++) {
                int xIndex = x + j;
                int yIndex = y + k;

                if (xIndex >= 0 && xIndex < width && yIndex >= 0 && yIndex < height) {
                    int filterIndex = (j + filterRadius) * filterDim + (k + filterRadius);
                    sum += inImage[(yIndex * width + xIndex) * channels + z] * filter[filterIndex];
                }
            }
        }

        opImage[i] = sum;
    }
}
