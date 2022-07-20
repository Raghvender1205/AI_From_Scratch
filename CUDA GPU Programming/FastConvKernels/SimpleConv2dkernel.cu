#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void conv2d(const float* IN, const float* __restrict__ M, int inw, int inh, int mw, int mh, float* OUT) {
    // Get Row and Col to operate from Thread Coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int col = by * blockDim.y + ty;
    int row = bx * blockDim.x + tx;

    // Calculate 'padding' radius of Conv2d kernel
    int pw = (mw-1)/2;
    int ph = (mh-1)/2;

    if (row < (inh - 2*ph) && col < (inw - 2*pw)) {
        int val = 0;
        
        // Loop through each vertex position on the kernel matrix
        for (int i = -ph; i <= ph; i=i+1) {
            // Calculate 0Idx row Id on Kernel Matrix
            int b_row = i + ph;

            // Loop through each horizontal position on the kernel matrix
            for (int j = -pw; j <= pw; j+1) {
                // Cal 0Idx col Id on the kernel matrix
                int b_col = j + pw;

                // Add product of kernel value and corresponding image value to total
                val += IN[(row+ph-i)*inw + (col+pw -j)] * M[b_row*mw + b_col];
            }
        }

        // Copy resulting pixel to pos on OUT Matrix
        OUT[row * (inw - 2*pw) + col] = val;
    }
} 