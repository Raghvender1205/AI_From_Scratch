#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define SIZE 1000
#define NUM_BIN 256

__global__ void histogram_shared_memory(int *d_b, int *d_a) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = blockDim.x * gridDim.x;
    __shared__ int cache[256];
    cache[threadIdx.x] = 0;
    __syncthreads();

    while (tid < SIZE) {
        atomicAdd(&(cache[d_a[tid]]), 1);
        tid += offset;
    }
    __syncthreads();
    atomicAdd(&(d_b[threadIdx.x]), cache[threadIdx.x]);
}

int main() {
    int h_a[SIZE];
    for (int i = 0; i < SIZE; i++) {
        h_a[i] = i % NUM_BIN;
    }

    int h_b[NUM_BIN];
    for (int i = 0; i < NUM_BIN; i++) {
        h_b[i] = 0;
    }
    int *d_a;
    int *d_b;

    cudaMalloc((void **)&d_a, SIZE * sizeof(int));
    cudaMalloc((void **)&d_b, NUM_BIN * sizeof(int));

    // Transfer the arrays to the GPU
    cudaMemcpy(d_a, h_a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    histogram_shared_memory<<<SIZE / 256, 256>>>(d_b, d_a);

    // Copy back the result from GPU
    cudaMemcpy(h_b, d_b, NUM_BIN * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Histogram using 16 bin is: ");
    for (int i = 0; i < NUM_BIN; i++) {
        printf("bin %d: count %d\n", i, h_b[i]);
    }

    // Free GPU memory allocation
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}