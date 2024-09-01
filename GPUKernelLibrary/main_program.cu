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

cv::Mat arrayToImage(float* imageArray, int rows, int cols, int channels) {
	try {
		cv::Mat image(rows, cols, CV_32FC3, imageArray);

		// Convert the float point to 8bit image
		cv::Mat outputImage;
		image.convertTo(outputImage, CV_8UC3, 255.0);

		return outputImage;
	}
	catch (cv::Exception& e) {
		const char* err_msg = e.what();
		std::cerr << "Exception caught: " << err_msg << std::endl;
		throw e;
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

	// Call kernel to add padding to imageArray with error handling

	cudaError_t cudaStatus;
	imagePaddingGPU <<< numBlocks, blockSize >>> (paddedImageArray, imageArray, rows, cols, channels, paddingSize);
	cudaDeviceSynchronize();

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "imagePaddingGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// Now that we have the padded image, we can perform 3x3 convolution using fixed filter
	// Define filter
	float h_filter[FILTER_SIZE * FILTER_SIZE] = {
		0.625f, 0.125f, 0.625f,
		0.125f, 0.25f, 0.125f,
		0.625f, 0.125f, 0.625f
	};

	float* d_filter;
	cudaMalloc(&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
	cudaMemcpy(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate memory for output image
	float* opImageArray;
	size_t opImageSize = image.rows * image.cols * image.channels() * sizeof(float);
	cudaMallocManaged(&opImageArray, opImageSize);

	int paddedWidth = cols + 2 * paddingSize;
	int paddedHeight = rows + 2 * paddingSize;

	// Call kernel to perform convolution
	convolution_RGB <<< numBlocks, blockSize >>> (paddedImageArray, opImageArray, d_filter, paddedWidth, paddedHeight, rows, cols, channels, FILTER_SIZE);
	cudaDeviceSynchronize();

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "convolution_RGB kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	// print some numbers from opimagearray

	for (int i = 0; i < 10; i++) {
		std::cout << opImageArray[i] << std::endl;
	}

	// Move opImageArray memory back to host
	// cudaMemPrefetchAsync(opImageArray, imageSize, cudaCpuDeviceId);

	// Convert opImageArray to opImage
	printf("Converting array to image... \n");

	cv::Mat outputImage = arrayToImage(opImageArray, rows, cols, channels);

	printf("Conversion complete. \n");
	printf("Saving image to root directory... \n");

	std::string filename = "output_image.jpg";
	bool saveStatus = cv::imwrite(filename, outputImage);

	printf("Image saved. \n");

	if (saveStatus) {
		std::cout << "Image successfully saved to the root directory as " << filename << std::endl;
	}
	else {
		std::cerr << "Failed to save the image." << std::endl;
	}

	cudaFree(d_filter);
	cudaFree(imageArray);
	cudaFree(paddedImageArray);
	cudaFree(opImageArray);

}

float array2Dto1D() {
	return 0.0f;
}