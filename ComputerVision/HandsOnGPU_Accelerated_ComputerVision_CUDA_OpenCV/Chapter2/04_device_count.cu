#include <memory>
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    // Number of CUDA enabled devices
    if (device_count == 0) {
        printf("There are no available device(s) supporting CUDA\n");
    } else {
        printf("Detected %d CUDA capable device(s)\n", device_count);
    }

    return 0;
}

// Detected 1 CUDA capable device(s)