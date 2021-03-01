#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <cuda.h>
#include <mma.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define M 16

#define M_TILES 1024

#define M_TOTAL (M * M_TILES)
#define WARP_SIZE 32
using namespace nvcuda;

__host__ void InitMatrix(half *A, half *B, half *C)
{
	for (int i = 0; i < M_TOTAL*M_TOTAL; i++)
		A[i] = __float2half(rand() % 1000 / 1000.0f);
	for (int i = 0; i < M_TOTAL*M_TOTAL; i++)
		B[i] = __float2half(rand() % 1000 / 1000.0f);
	for (int i = 0; i < M_TOTAL*M_TOTAL; i++)
		C[i] = __float2half(rand() % 1000 / 1000.0f);
}

__global__ void fma_tensor(half *A, half *B, half *C, half *D)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x)/WARP_SIZE;
	int iy = (blockIdx.y * blockDim.y + threadIdx.y);

	wmma::fragment<wmma::matrix_a, M, M, M, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, M, M, M, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, M, M, M, half> ab_frag;
	wmma::fragment<wmma::accumulator, M, M, M, half> c_frag;
	
	wmma::fill_fragment(ab_frag, __float2half(0.0f));

	// AB = A*B
	int a_col, a_row, b_col, b_row, c_col, c_row;
	a_row = ix * M;
	b_row = iy * M;
	for (int k=0; k<M_TOTAL; k+=M) {
		a_col = b_col = k;

		if (a_row < M_TOTAL && a_col < M_TOTAL && b_row < M_TOTAL && b_col < M_TOTAL) {
			// Load the inputs
			wmma::load_matrix_sync(a_frag, A + a_col + a_row * M_TOTAL, M_TOTAL);
			wmma::load_matrix_sync(b_frag, B + b_col + b_col * M_TOTAL, M_TOTAL);

			// Perform the matrix multiplication
			wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
		}
	}

	// D = AB + C
	c_col = b_row;
	c_row = a_row;
	if (c_row < M_TOTAL && c_col < M_TOTAL) {
		wmma::load_matrix_sync(c_frag, C + c_col + c_row * M_TOTAL, M_TOTAL, wmma::mem_row_major);

		for (int i = 0; i < c_frag.num_elements; i++) {
			c_frag.x[i] = ab_frag.x[i] + c_frag.x[i];
		}

		// Store the output
		wmma::store_matrix_sync(D + c_col + c_row * M_TOTAL, c_frag, M_TOTAL, wmma::mem_row_major);
	}
}


int main()
{
	half *h_A, *h_B, *d_A, *d_B;
	half *h_C, *h_D, *d_C, *d_D;

	h_A = new half[M_TOTAL*M_TOTAL];
	h_B = new half[M_TOTAL*M_TOTAL];
	h_C = new half[M_TOTAL*M_TOTAL];
	h_D = new half[M_TOTAL*M_TOTAL];
	
	InitMatrix(h_A, h_B, h_C);
	printf("Matrices are size:\n");
	printf("h_A: %d x %d\n", M_TOTAL, M_TOTAL);	
	printf("h_B: %d x %d\n", M_TOTAL, M_TOTAL);	
	printf("h_C: %d x %d\n", M_TOTAL, M_TOTAL);	
	int MSizeBytesHalf;
	
	MSizeBytesHalf = sizeof(half) * M_TOTAL * M_TOTAL;

	cudaMalloc((void**)&d_A, MSizeBytesHalf);
	cudaMalloc((void**)&d_B, MSizeBytesHalf);
	cudaMalloc((void**)&d_C, MSizeBytesHalf);
	cudaMalloc((void**)&d_D, MSizeBytesHalf);

	cudaMemcpy(d_A, h_A, MSizeBytesHalf, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_B, h_B, MSizeBytesHalf, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_C, h_C, MSizeBytesHalf, cudaMemcpyHostToDevice); 

	//Kernel Invoke Paramters (2D grid and blocks) 
	int dimx = 16; 
	int dimy = 16; 

	dim3 block(dimx, dimy); //Block of 256 threads 
	dim3 grid((M_TOTAL+block.x-1)/block.x, (M_TOTAL+block.y-1)/block.y); //grid dimensions 

	printf("Value of block %d \t %d \n",block.x,block.y);
	printf("Value of grid %d \t %d \n",grid.x,grid.y);

	// Record time using CUDA events
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	fma_tensor<<<block, grid>>>(d_A,d_B,d_C,d_D);
	cudaEventRecord(stop);
	cudaDeviceSynchronize(); 
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Tensor FMA execution took %f ms \n", milliseconds);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaMemcpy(h_D, d_D, MSizeBytesHalf, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_D);

	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	delete[] h_D;

return 0;
}
