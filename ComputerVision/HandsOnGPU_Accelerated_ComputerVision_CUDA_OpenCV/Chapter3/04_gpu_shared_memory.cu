#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void gpu_shared_memory(float *d_a) {
    int i, idx = threadIdx.x;
    float avg, sum = 0.0f;

    // Define shared memory
    __shared__ float sh_arr[10];

    sh_arr[idx] = d_a[idx];
    __syncthreads(); // Ensures all the writes to shared memory have completed

    for (int i = 0; i <= idx; i++) {
        sum += sh_arr[i];
    }
    avg = sum / (idx + 1.0f);

    d_a[idx] = avg;
    sh_arr[idx] = avg;
}

int main() {
    float h_a[10];
    float *d_a;

    for (int i = 0; i < 10; i++) {
		h_a[i] = i;
	}
	// Allocate global memory on the device
	cudaMalloc((void **)&d_a, sizeof(float) * 10);
	// Copy data from host memory  to device memory 
	cudaMemcpy((void *)d_a, (void *)h_a, sizeof(float) * 10, cudaMemcpyHostToDevice);
	
	gpu_shared_memory << <1, 10 >> >(d_a);
	// Copy the modified array back to the host memory
	cudaMemcpy((void *)h_a, (void *)d_a, sizeof(float) * 10, cudaMemcpyDeviceToHost);
	
    printf("Use of Shared Memory on GPU:  \n");
	for (int i = 0; i < 10; i++) {
		printf("The running average after %d element is %f \n", i, h_a[i]);
	}
	
    return 0;
}

/**
Use of Shared Memory on GPU:  
The running average after 0 element is 0.000000 
The running average after 1 element is 0.500000
The running average after 2 element is 1.000000
The running average after 3 element is 1.500000
The running average after 4 element is 2.000000
The running average after 5 element is 2.500000
The running average after 6 element is 3.000000
The running average after 7 element is 3.500000
The running average after 8 element is 4.000000
The running average after 9 element is 4.500000
*/