#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "imagePadding.cuh"

using namespace std;

__global__ void imagePaddingGPU(float* paddedImageArray, float* imageArray, int rows, int cols, int channels, int paddingSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = (rows + 2 * paddingSize) * (cols + 2 * paddingSize) * channels;

    if (x >= totalElements) return;

    int z = x % channels;
    int y = (x / channels) % (cols + 2 * paddingSize);
    int row = x / ((cols + 2 * paddingSize) * channels);

    if (row < paddingSize || row >= (rows + paddingSize) || y < paddingSize || y >= (cols + paddingSize)) {
        // Padding pixels, replicate border pixels
        int origRow = min(max(row - paddingSize, 0), rows - 1);
        int origCol = min(max(y - paddingSize, 0), cols - 1);
        paddedImageArray[x] = imageArray[(origRow * cols + origCol) * channels + z];
    }
    else {
        // Inner pixels
        int origRow = row - paddingSize;
        int origCol = y - paddingSize;
        paddedImageArray[x] = imageArray[(origRow * cols + origCol) * channels + z];
    }
}