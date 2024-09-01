#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "2DConvolution_RGB.cuh"

__global__ void convolution_RGB(float *inImage, float *opImage, float filter, int width, int height, int filterDim) {

}