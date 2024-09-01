# Convolution Library for Images
The program adds padding to an image and then applies convolution using a configurable filter, all in parallel using CUDA. The default filter will apply a Gaussian Blur.

# Other kernels to test with (sobel filters)
Brighten image matrix:
0.2  0.2  0.2
0.2  0.2  0.2
0.2  0.2  0.2

Darken image:
0.1  0.1  0.1
0.1  0.1  0.1
0.1  0.1  0.1

Gausian Blur:
0.0625 0.125 0.0625
0.125 0.25 0.125
0.0625 0.125 0.0625
