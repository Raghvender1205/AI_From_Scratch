#include <iostream>
#include <stdio.h>

__global__ void firstKernel(void) {

}

int main(void) {
    firstKernel<<<1, 1>>>();
    printf("Hello, CUDA\n");
    return 0;
}