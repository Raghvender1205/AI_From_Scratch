#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) {
    *d_c = *d_a + *d_b;
}

int main() {
    int h_a, h_b, h_c;
    int *d_a, *d_b, *d_c;

    h_a = 1;
    h_b = 4;

    cudaError_t cudaStatus;
    // Allocate GPU buffers for 3 vectors
    cudaStatus = cudaMalloc((void**)&d_c, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    cudaStatus = cudaMalloc((void**)&d_a, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_b, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy Input vectors from Host memory to GPU buffers
    cudaStatus = cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch Kernel on GPU with 1 Thread per element
    gpuAdd<<<1, 1>>>(d_a, d_b, d_c);

    // Check for any errors launching Kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy output Vector from GPU buffer to host memory
    cudaStatus = cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    printf("Passing Parameter by Reference Output: %d + %d = %d\n", h_a, h_b, h_c);
Error:
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;

}