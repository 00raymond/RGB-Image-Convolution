#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "matrixmultiplykernel.cuh"
#include "2DConvolution_RGB.cuh"
#include "imagePadding.cuh"

#define FILTER_SIZE 3

void populateImageArray(cv::Mat image, float* imageArray) {
	int rows = image.rows;
	int cols = image.cols;
	int channels = image.channels();

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < channels; k++) {
				// Normalize pixel value to [0.0, 1.0] and store as float
				imageArray[i * cols * channels + j * channels + k] = image.at<cv::Vec3b>(i, j)[k] / 255.0f;
			}
		}
	}
}

int main() {

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		std::cerr << "No CUDA-compatible devices found!" << std::endl;
		return -1;
	}

	int deviceId;
	cudaGetDevice(&deviceId);

	// Load image
	// get image from path assets/Sample-jpg-image.jpg
	std::string imagePath = "test.jpg";
	cv::Mat image = cv::imread(imagePath);

	if (image.empty()) {
		std::cerr << "Could not open or find the image!" << std::endl;
		return -1;
	}

	float* imageArray;
	size_t imageSize = image.rows * image.cols * image.channels() * sizeof(float);
	cudaMallocManaged(&imageArray, imageSize);
	populateImageArray(image, imageArray);

	// Asynchronous memory pre fetching for image array
	// Call kernel to add replication padding to imageArray
	int paddingSize = FILTER_SIZE / 2;

	// Updated row/col/channels variables for padded image
	int rows = image.rows;
	int cols = image.cols;
	int channels = image.channels();

	int blockSize = 256;
	int numBlocks = ((rows * cols * channels) + blockSize - 1) / blockSize;

	// Allocate memory for padded image
	size_t paddedImageSize = (rows + 2 * paddingSize) * (cols + 2 * paddingSize) * channels * sizeof(float);
	float* paddedImageArray;

	// Asynchronous memory pre fetching for image arrays
	cudaMallocManaged(&paddedImageArray, paddedImageSize);

	/*cudaMemPrefetchAsync(paddedImageArray, paddedImageSize, deviceId);
	cudaMemPrefetchAsync(imageArray, imageSize, deviceId);*/

	printf("here");

	// Call kernel to add padding to imageArray with error handling

	cudaError_t cudaStatus;
	imagePaddingGPU <<< numBlocks, blockSize >>> (paddedImageArray, imageArray, rows, cols, channels, paddingSize);
	cudaDeviceSynchronize();

	// print items from arrays to see if they are the same
	for (int i = 100; i < 120; i++) {
		std::cout << "imageArray[" << i << "] = " << imageArray[i] << std::endl;
		std::cout << "paddedImageArray[" << i << "] = " << paddedImageArray[i] << std::endl;
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "imagePaddingGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// Now that we have the padded image, we can perform 3x3 convolution using fixed filter
	// Define filter
	float filter[FILTER_SIZE * FILTER_SIZE] = {
		0.1111f, 0.4111f, 0.2111f,
		0.5711f, 0.1211f, 0.1311f,
		0.1831f, 0.1211f, 0.3614f
	};

	// Allocate memory for output image
	float* opImageArray;
	cudaMallocManaged(&opImageArray, imageSize);

	// Call kernel to perform convolution
	convolution_RGB <<< numBlocks, blockSize >>> (paddedImageArray, opImageArray, filter, rows, cols, FILTER_SIZE);


	cudaFree(imageArray);
	cudaFree(paddedImageArray);

}

float array2Dto1D() {
	return 0.0f;
}