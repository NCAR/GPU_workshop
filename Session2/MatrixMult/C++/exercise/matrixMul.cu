#include "pch.h"

__global__ void mmul( float *A, float *B, float *C, int m, int p, int q)
{
   //Calculate the row and column values based on the block Id, block dimensions and the thread Id.
 
   //Multiply Matrices A and B, store results in Matrix C

}



__host__ void gpuMult(float *h_A, float *h_B, float *gpu_C, const int m, const int p, const int q)
{
  //declare variables to be used by GPU (device) for matrix multiplication
  float *d_A, *d_B, *d_C;

  //Allocate device memory
  //cudaMalloc(&d_A, ???*???*sizeof(???));
  //cudaMalloc(&d_B, ???*???*sizeof(???));
  //cudaMalloc(&d_C, ???*???*sizeof(???));
  cudaCheckErrors("cudaMalloc failure");

  // Copy host matrices A and B to the device using cudaMemcpy
  //cudaMemcpy(dest, src, ???*???*sizeof(???), cudaMemcpyHostToDevice);
  //cudaMemcpy(dest, src, ???*???*sizeof(???), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failture");
  
  // Set block dimensions here
  // Remember: the maximum number of total threads is 1024.
  unsigned int block_size = BLOCK_SIZE; // from pch.h is 32
  dim3 block(block_size, block_size);
  //calculate grid dimensions here
  //unsigned int grid_rows = ???; 
  //unsigned int grid_cols = ???; 
  dim3 grid(grid_cols, grid_rows);
 
  printf("Kernel launch dimensions: \n");
  printf("\tGrid size  : {%d, %d, %d} blocks.\n",grid.x, grid.y, grid.z);
  printf("\tBlock size : {%d, %d, %d} threads.\n",block.x, block.y, block.z);

  //Launch matrix multiplication kernel (the global function)

  // block CPU until GPU returns data using cudaDeviceSynchronize 

  // Transfer results from device to host 

  cudaCheckErrors("Kernel execution failure or cudaMemcpy H2D failure");

  // Cleanup - free memory on GPU using cudaFree



  cudaCheckErrors("cudaFree failure");
}
