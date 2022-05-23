#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 5

__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) {
	// Getting Thread index of current kernel
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N) {
		d_c[tid] = d_a[tid] + d_b[tid];
		tid += blockDim.x * gridDim.x;
	}
}

int main() {
    int h_a[N], h_b[N], h_c[N];
    int *d_a, *d_b, *d_c;

    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
    cudaEventRecord(e_start, 0);

    // Allocate the memory
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_b, N * sizeof(int));
    cudaMalloc((void **)&d_c, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_a[i] = 2 * i*i;
        h_b[i] = i;
    }

    // Copy input arrays from host to device memory
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
    // GPU Kernel Call
    gpuAdd << <512, 512>> >(d_a, d_b, d_c);

    // Copy result back to host memory from device
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(e_stop);
    cudaEventSynchronize(e_stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    printf("Time to add %d numbers: %3.1f ms\n", N, elapsedTime);

    int correct = 1;
    printf("Vector Addition on GPU\n");
    for (int i = 0; i < N; i++) {
        if ((h_a[i] + h_b[i] != h_c[i])) {
            correct = 0;
        }
    }

    if (correct == 1) {
        printf("GPU has computed sum Correctly\n");
    } else {
        printf("There is an Error in GPU computation\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

/**
Time to add 5 numbers: 1.9 ms
Vector Addition on GPU
GPU has computed sum Correctly
*/