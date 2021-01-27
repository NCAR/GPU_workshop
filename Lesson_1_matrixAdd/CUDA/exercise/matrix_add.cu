#include <cuda_runtime.h>
#include <cuda.h>
#include "pch.h"
#include <stdio.h>
 
//Add the keyword to make it a GPU function
 void add_matrices(float* m1, float* m2, float* sum, int m, int n){

    // Calculate the thread idx and idy index 
    

    // Calculate the Addition


}

//Add the keyword to make it a host function
 void gpu_matrix_add(const float *h_A, const float *h_B, float *h_C,\
   const int dx, const int dy)
{
  float *d_A, *d_B, *d_C;

  // Allocate memory on GPU for the matrices

  // Check for any errors
  cudaCheckErrors("cudaMalloc failure");

  // Copy the matrices A, B from host to device



  cudaCheckErrors("cudaMemcpy H2D failture");

  // Kernel Invoke Paramters (2D grid and blocks) 
  // Decide the dimensions of a block
  int dimx = 16; 
  int dimy = 16; 
  // Block of 256 threads 
  dim3 block(dimx, dimy); 

  /* Calculate the grid dimensions based on the dimensions of the matrix,
    and dimensions of the block */
  dim3 grid((dx+block.x-1)/block.x, (dy+block.y-1)/block.y); 

  printf("Kernel launch dimensions: \n");
  printf("\tGrid size  : {%d, %d, %d} blocks.\n",grid.x, grid.y, grid.z);
  printf("\tBlock size : {%d, %d, %d} threads.\n",block.x, block.y, block.z);
  
  // Calcuate A+B=C on the GPU


  cudaCheckErrors("kernel launch failure");

  // Block the CPU until GPU finishes execution


  // Copy the result matrix C from device to host

  cudaCheckErrors("Kernel execution failure or cudaMemcpy H2D failure");

  // deallocate memory on GPU
  cudaCheckErrors("cudaFree failure");

  // Reset Device
  cudaDeviceReset();
}
