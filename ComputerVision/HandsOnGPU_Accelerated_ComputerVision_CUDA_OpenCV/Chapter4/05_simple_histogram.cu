#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 1000
#define NUM_BIN 16

__global__ void histogram_without_atomic(int *d_b, int *d_a) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int item = d_a[tid];
    if (tid < SIZE) {
        d_b[item]++;
    }
}

__global__ void histogram_atomic(int *d_b, int *d_a) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int item = d_a[tid];
    if (tid < SIZE) {
        atomicAdd(&(d_b[item]), 1);
    }
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

    // declare GPU memory pointers
    int *d_a;
    int *d_b;

    // allocate GPU memory
    cudaMalloc((void **)&d_a, SIZE * sizeof(int));
    cudaMalloc((void **)&d_b, NUM_BIN * sizeof(int));

    // transfer the arrays to the GPU
    cudaMemcpy(d_a, h_a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel

    histogram_without_atomic<<<((SIZE + NUM_BIN - 1) / NUM_BIN), NUM_BIN>>>(d_b, d_a);
    // histogram_atomic << <((SIZE+NUM_BIN-1) / NUM_BIN), NUM_BIN >> >(d_b, d_a);

    // copy back the sum from GPU
    cudaMemcpy(h_b, d_b, NUM_BIN * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Histogram using 16 bin without shared Memory is: \n");
    for (int i = 0; i < NUM_BIN; i++) {
        printf("bin %d: count %d\n", i, h_b[i]);
    }

    // Free GPU memory allocation
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}

/**
Histogram using 16 bin without shared Memory is:
bin 0: count 1
bin 1: count 1
bin 2: count 1
bin 3: count 1
bin 4: count 1
bin 5: count 1
bin 6: count 1
bin 7: count 1
bin 8: count 1
bin 9: count 1
bin 10: count 1
bin 11: count 1
bin 12: count 1
bin 13: count 1
bin 14: count 1
bin 15: count 1
*/