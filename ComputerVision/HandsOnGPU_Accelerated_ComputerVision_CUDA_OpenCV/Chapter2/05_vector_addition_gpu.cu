#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 5

// Kernel for Vector Addition
__global__ void gpuAdd(int* d_a, int* d_b, int* d_c) {
    // Get Block Idx of current Kernel
    int tid = blockIdx.x; 
    // Handle data at idx    
    if (tid < N) {
        d_c[tid] = d_a[tid] + d_b[tid];
    }
}

int main(void) {
    int h_a[N], h_b[N], h_c[N]; // Host Arrays
    int *d_a, *d_b, *d_c; // Device Arrays

    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    // Init Arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = 2 * i * i;
        h_b[i] = i;
    }

    // Copy input arrays from host to device memory
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Calling kernels with N blocks and 1 thread per block, passing device ptrs as params
    gpuAdd << <N, 1>> >(d_a, d_b, d_c);
    // Copy result back to host memory from device memory
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Vector addition on GPU \n");
    for (int i = 0; i < N; i++) {
		printf("The sum of %d element is %d + %d = %d\n", i, h_a[i], h_b[i], h_c[i]);
	}
	// Free up memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}