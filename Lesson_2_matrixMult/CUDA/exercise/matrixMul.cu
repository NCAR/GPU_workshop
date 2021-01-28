#include "pch.h"

__global__ void mmul( float *a, float *b, float *c, int m, int n, int q)
{
   //Calculate the row and column values based on the block Id, block dimensions and the thread Id.
 
   //Multiply Matrices A and B, store results in Matrix C

}



__host__ void gpuMult(float *h_A, float *h_B, float *gpu_C, const int m, const int n, const int p, const int q, const int block_size)
{
  //declare variables to be used by GPU (device) for matrix multiplication
  float *d_A, *d_B, *d_C;

  //Allocate device memory
  cudaMalloc(&d_A, m*n*sizeof(float));
  cudaMalloc(&d_B, p*q*sizeof(float));
  cudaMalloc(&d_C, m*q*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, m*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, p*q*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failture");
  
  //calculate grid and block dimensions here
   

 
  printf("Kernel launch dimensions: \n");
  printf("\tGrid size  : {%d, %d, %d} blocks.\n",grid.x, grid.y, grid.z);
  printf("\tBlock size : {%d, %d, %d} threads.\n",block.x, block.y, block.z);


  //Launch matrix multiplication kernel, and block CPU until GPU returns data



  cudaCheckErrors("Kernel execution failure or cudaMemcpy H2D failure");
  

  // Transfer results from device to host 

  // Cleanup - free memory on GPU

  cudaCheckErrors("cudaFree failure");
}
