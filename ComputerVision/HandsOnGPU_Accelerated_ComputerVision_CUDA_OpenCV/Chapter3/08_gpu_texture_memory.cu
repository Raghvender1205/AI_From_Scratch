#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREADS 10
#define N 10

texture <float, 1, cudaReadModeElementType> textureRef;
__global__ void gpu_texture_memory(int n, float* d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float temp = tex1D(textureRef, float(idx));
        d_out[idx] = temp;
    }
}

int main() {
    int num_blocks = N / NUM_THREADS + ((N % NUM_THREADS) ? 1 : 0);
    float* d_out;

    // Allocate space on the device
    cudaMalloc((void**)&d_out, sizeof(float) * N);
    // Allocate space on host for results
    float *h_out = (float*)malloc(sizeof(float)* N);
    float h_in[N];
 
    for (int i = 0; i < N; i++) {
        h_in[i] = float(i);
    }
    
    // CUDA Array
    cudaArray *cu_arr;
    cudaMallocArray(&cu_arr, &textureRef.channelDesc, N, 1);
    // Copy data to CUDA Array
    cudaMemcpyToArray(cu_arr, 0, 0, h_in, sizeof(float)*N, cudaMemcpyHostToDevice);

    // Bind a texture to CUDA Array
    cudaBindTextureToArray(textureRef, cu_arr);
    
    gpu_texture_memory << <num_blocks, NUM_THREADS>> >(N, d_out);

    // Copy result back to Host
    cudaMemcpy(h_out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost);
    printf("Use of Texture Memory on GPU\n");
    for (int i = 0; i < N; i++) {
        printf("Texture element at %d is: %f\n", i, h_out[i]);
    }
    free(h_out);
    cudaFree(d_out);
    cudaFreeArray(cu_arr);
    cudaUnbindTexture(textureRef); // Unbin the Texture 
}   

/**
Use of Texture Memory on GPU
Texture element at 0 is: 0.000000
Texture element at 1 is: 1.000000
Texture element at 2 is: 2.000000
Texture element at 3 is: 3.000000
Texture element at 4 is: 4.000000
Texture element at 5 is: 5.000000
Texture element at 6 is: 6.000000
Texture element at 7 is: 7.000000
Texture element at 8 is: 8.000000
Texture element at 9 is: 9.000000
*/