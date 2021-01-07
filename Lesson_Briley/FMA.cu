//#include "Wkshp_head.h"
#include <cuda_runtime.h>
#include <iostream> 
#include <stdio.h>
#include <cuda.h> 

__global__ void NaiveMult(float *d_A, float *d_B, float *d_C, const int ny, const int nx)
{
        int row = threadIdx.y+(blockIdx.y*blockDim.y);
        int col = threadIdx.x +(blockIdx.x*blockDim.x);
        float fSum = 0.0f;

        if (row<ny && col<nx) {
                for(int k=0; k<nx; k++)
                {
                        fSum += d_A[row*nx+k]*d_B[k*nx +col];
                }
                d_C[row*nx+col] = fSum;
        }

}
 
__host__ void gpuMult(float *h_A, float *h_B, float *h_C, const int ny, const int nx)
{ 
	float *d_A, *d_B, *d_C; 
	const int MSizeBytes = ny*nx*sizeof(float); 

	//Allocate memory on device 
	cudaMalloc((void**)&d_A, MSizeBytes);		
	cudaMalloc((void**)&d_B, MSizeBytes); 
	cudaMalloc((void**)&d_C, MSizeBytes); 

	//Copy input data to device 
	cudaMemcpy(d_A, h_A, MSizeBytes, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_B, h_B, MSizeBytes, cudaMemcpyHostToDevice);

	//Kernel Invoke Paramters (2D grid and blocks) 
	int dimx = 16; 
	int dimy = 16; 

	dim3 block(dimx, dimy); //Block of 256 threads 
	dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y); //grid dimensions 

	//Multiplication 
	NaiveMult<< <grid, block>> >(d_A,d_B,d_C,ny,nx);	
	
	cudaDeviceSynchronize(); 
	
	//Copy Results back 
	cudaMemcpy(h_C, d_C, MSizeBytes, cudaMemcpyDeviceToHost);
 
	//Memory Release 
	cudaFree(d_A); 
	cudaFree(d_B); 
	cudaFree(d_B);
	cudaDeviceReset(); 	

}


