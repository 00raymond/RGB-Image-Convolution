#ifndef MATMUL_CUH
#define MATMUL_CUH

#include "cuda_runtime.h"

__global__ void matrixMulGPU(int* a, int* b, int* c);

#endif
