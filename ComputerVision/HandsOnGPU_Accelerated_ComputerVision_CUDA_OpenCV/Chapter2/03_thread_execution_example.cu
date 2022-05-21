#include <iostream>

#include <stdio.h>
__global__ void kernel(void) {
    printf("Hello!! thread in block: %d\n", blockIdx.x);
}

int main(void) {
    kernel << <16, 1>> >(); // Kernel call with 16 blocks and 1 thread per block
     
    // Wait for all Kernels to finish
    cudaDeviceSynchronize();
    printf("All Threads are finished");
    return 0;
}

/**
Hello!! thread in block: 7
Hello!! thread in block: 4
Hello!! thread in block: 13
Hello!! thread in block: 1
Hello!! thread in block: 10
Hello!! thread in block: 6
Hello!! thread in block: 8
Hello!! thread in block: 3
Hello!! thread in block: 5
Hello!! thread in block: 12
Hello!! thread in block: 14
Hello!! thread in block: 0
Hello!! thread in block: 2
Hello!! thread in block: 9
Hello!! thread in block: 11
Hello!! thread in block: 15
All Threads are finished
*/