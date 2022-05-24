#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define arraySize 100
#define ThreadPerBlock 5

__global__ void addKernel(int *d_a, int *d_b) {
    int cnt = 0;
    int tid = threadIdx.x;
    int ttid = blockIdx.x * ThreadPerBlock + tid;
    int val = d_a[ttid];

    __shared__ int cache[ThreadPerBlock];
    for (int i = tid; i < arraySize; i += ThreadPerBlock) {
        cache[tid] = d_a[i];
        __syncthreads();
        for (int j = 0; j < ThreadPerBlock; j++) {
            if (val > cache[j]) {
                cnt++;
            }
        }
        __syncthreads();
    }
    d_b[cnt] = val;
}

int main() {
    int h_a[arraySize] = {5, 9, 3, 4, 8};
    int h_b[arraySize];
    int *d_a, *d_b;

    cudaMalloc((void **)&d_b, arraySize * sizeof(int));
    cudaMalloc((void **)&d_a, arraySize * sizeof(int));

    // Copy input vector from host memory to GPU buffers.
    cudaMemcpy(d_a, h_a, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<arraySize / ThreadPerBlock, ThreadPerBlock>>>(d_a, d_b);

    cudaDeviceSynchronize();
    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(h_b, d_b, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    printf("The Enumeration sorted Array is: \n");
    for (int i = 0; i < arraySize; i++) {
        printf("%d\n", h_b[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}