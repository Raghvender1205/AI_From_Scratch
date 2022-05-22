#include <stdio.h>
#define N 5

__global__ void gpu_local_memory(int d_in) {
    int t_local;
    t_local = d_in * threadIdx.x;

    printf("Value of Local Variable in Current Thread: %d\n", t_local);
}

int main() {
    printf("Use of Local Memory on GPU:\n");
    gpu_local_memory << <1, N>> >(5);
    cudaDeviceSynchronize();

    return 0;
}

/**
Use of Local Memory on GPU:
Value of Local Variable in Current Thread: 0
Value of Local Variable in Current Thread: 5
Value of Local Variable in Current Thread: 10
Value of Local Variable in Current Thread: 15
Value of Local Variable in Current Thread: 20
*/