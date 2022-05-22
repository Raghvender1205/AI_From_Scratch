#include <stdio.h>
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 5

// Kernel for Vector Addition
__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) {
	// Getting block index of current kernel
	int tid = threadIdx.x + blockIdx.x * blockDim.x;	
	
    while (tid < N) {
		d_c[tid] = d_a[tid] + d_b[tid];
		tid += blockDim.x * gridDim.x;
	}	
}

int main() {
    int h_a[N], h_b[N], h_c[N]; // Host Arrays
    int *d_a, *d_b, *d_c; // device ptr

    cudaMalloc((void**)&d_a, N * sizeof(int));
	cudaMalloc((void**)&d_b, N * sizeof(int));
	cudaMalloc((void**)&d_c, N * sizeof(int));
	// Initializing Arrays
	for (int i = 0; i < N; i++) {
		h_a[i] = 2 * i*i;
		h_b[i] = i;
	}

    // Copy input arr from host to device memory
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Call Kernels for with N blocks and 1 thread per block, pass device ptr as params
    gpuAdd << <512, 512>> >(d_a, d_b, d_c);
    // Copy result back to host memory from device memory
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    int correct = 1;
    printf("Vector Addition on GPU\n");
    for (int i = 0; i < N; i++) {
        if ((h_a[i] + h_b[i] != h_c[i])) {
            correct = 0;
        }
    }

    if (correct == 1) {
        printf("GPU has computed sum correctly");
    } else {
        printf("There is an Error in GPU computation");
    }

    // CUDA Free Memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}