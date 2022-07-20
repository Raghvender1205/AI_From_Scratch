#include <cuda_runtime.h>
#include <stdio.h>

#define MASK_WIDTH 5
#define INPUT_SIZE 12
#define TILE_SIZE 4
__constant__ float M[MASK_WIDTH];

__global__ void convolution_shared_memory(float* N, float* P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float N_s[TILE_SIZE];
    N_s[threadIdx.x] = N[i];
    __syncthreads();
    
    int this_title_start_point = blockIdx.x * blockDim.x;
    int next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
    float Pvalue = 0;
    int n_start_point = i - (MASK_WIDTH / 2);

    for (int j = 0; j < MASK_WIDTH; j++) {
        int N_index = n_start_point + j;

        if (N_index >= 0 && N_index < INPUT_SIZE) {
            if ((N_index >= this_title_start_point) && (N_index < next_tile_start_point)) {
                Pvalue += N_s[threadIdx.x + j - (MASK_WIDTH/2)] * M[j];
            } else {
                Pvalue += N[N_index] * M[j];
            }
        }
    }
    P[i] = Pvalue;
}

__global__ void convolution_constant_memory(float *N, float *P, int Width) {
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	float Pvalue = 0;
	int n_start_point = i-(MASK_WIDTH/2);

	for(int j =0; j<MASK_WIDTH;j++){
		if(n_start_point+j >=0 && n_start_point+j < Width){
			Pvalue+= N[n_start_point+j]*M[j];
		}
	}
	P[i]=Pvalue;
}

__global__ void convolution_global_memory(float* N, float* M, float* P, int Width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int n_start_point = i - (MASK_WIDTH / 2);

    for (int j = 0; j < MASK_WIDTH; j++) {
        if (n_start_point + j >= 0 && n_start_point + j < Width) {
            Pvalue += N[n_start_point + j] * M[j];
        }
    }

    P[i] = Pvalue;
}

int main() {
    // device input and output
    float *d_N = 0;
	float *d_P = 0;

    cudaMalloc(&d_N, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_P,INPUT_SIZE*sizeof(float));

	//host input and output
	float *h_N = (float*)malloc(INPUT_SIZE*sizeof(float));
	float *h_P = (float*)malloc(INPUT_SIZE*sizeof(float));
	float *h_M = (float*)malloc(MASK_WIDTH*sizeof(float));

	//initialize input on host
	for(int i=0;i<INPUT_SIZE;++i){
		h_N[i]=(float)i;
	}

	//transfer input to device
	cudaMemcpy(d_N,h_N,INPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_P,h_P,INPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice);

	//initialize mask on host
	for(int j=0;j<MASK_WIDTH;++j){
		h_M[j]=(float)j;
	}

	//transfer mask to constant memory
	cudaMemcpyToSymbol(M,h_M,MASK_WIDTH*sizeof(float));


	//call convolution kernel
	convolution_shared_memory<<<(INPUT_SIZE+TILE_SIZE-1)/TILE_SIZE,TILE_SIZE >>>(d_N,d_P);

	//retrieve result from device
	cudaMemcpy(h_P,d_P,INPUT_SIZE*sizeof(float),cudaMemcpyDeviceToHost);

	for(int i=0; i<INPUT_SIZE;++i){
		printf("%f\n", h_P[i]);
	}

	cudaFree(d_N);
	cudaFree(d_P);
	cudaFree(M);

	free(h_N);
	free(h_P);
	free(h_M);
}