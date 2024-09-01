#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matrixmultiplykernel.cuh"

#define N  64
#define I2D(n, r, c) ((r*n) + c)

__global__ void matrixMulGPU(int* a, int* b, int* c)
{

    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y;

    // since we have row and column, for each element in row and column, add their respective positions.
    int addValToC = 0;
    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            addValToC += a[I2D(N, row, k)] * b[I2D(N, k, col)];
        }
        c[(row * N) + col] = addValToC;
    }

}
