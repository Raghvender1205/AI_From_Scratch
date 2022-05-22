#include <stdio.h>

#define NUM_THREADS 10000
#define SIZE  10

#define BLOCK_WIDTH 100

__global__ void gpu_increment_atomic(int *d_a){
	// Calculate thread id for current thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread increments elements wrapping at SIZE variable
	tid = tid % SIZE;
	atomicAdd(&d_a[tid], 1);
}

int main(int argc, char **argv) {
	printf("%d total threads in %d blocks writing into %d array elements\n",
		NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, SIZE);

	// Declare and allocate host memory
	int h_a[SIZE];
	const int ARRAY_BYTES = SIZE * sizeof(int);

	// Declare and allocate GPU memory
	int * d_a;
	cudaMalloc((void **)&d_a, ARRAY_BYTES);
	// Initialize GPU memory to zero
	cudaMemset((void *)d_a, 0, ARRAY_BYTES);
	
	gpu_increment_atomic << <NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH >> >(d_a);
	

	// Copy back the array to host memory
	cudaMemcpy(h_a, d_a, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	
	printf("Number of times a particular Array index has been incremented is: \n");
	for (int i = 0; i < SIZE; i++)  { 
		printf("index: %d --> %d times\n ", i, h_a[i]); 
	}
	
	cudaFree(d_a);
	return 0;
}

/**
10000 total threads in 100 blocks writing into 10 array elements
Number of times a particular Array index has been incremented is: 
index: 0 --> 1000 times
 index: 1 --> 1000 times
 index: 2 --> 1000 times
 index: 3 --> 1000 times
 index: 4 --> 1000 times
 index: 5 --> 1000 times
 index: 6 --> 1000 times
 index: 7 --> 1000 times
 index: 8 --> 1000 times
 index: 9 --> 1000 times
*/