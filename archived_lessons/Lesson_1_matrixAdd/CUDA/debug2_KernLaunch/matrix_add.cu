#include <cuda_runtime.h>
#include <cuda.h>
#include "pch.h"
#include <stdio.h>
 
__global__ void add_matrices(float* m1, float* m2, float* sum, int m, int n){

    // Calculate the thread x and y index 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate thread id
    int index = idx*n+idy; 
    if (idx < m && idy < n) sum[index] = m1[index]+m2[index];
}

__host__ void gpu_matrix_add(const float *h_A, const float *h_B, float *h_C,\
   const int dx, const int dy)
{
  float *d_A, *d_B, *d_C;

  // Allocate memory on GPU to matrices 
  cudaMalloc(&d_A, dx*dy*sizeof(float));
  cudaMalloc(&d_B, dx*dy*sizeof(float));
  cudaMalloc(&d_C, dx*dy*sizeof(float));

  // Check for any errors
  cudaCheckErrors("cudaMalloc failure");

  // Copy the matrices A, B from host to device
  cudaMemcpy(d_A, h_A, dx*dy*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, dx*dy*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failture");

  // Kernel Invoke Paramters (2D grid and blocks) 
  // Decide the dimensions of a block
  int dimx = 256; 
  int dimy = 256; 
  // Block of 256 threads 
  dim3 block(dimx, dimy); 

  /* Calculate the grid dimensions based on the dimensions of the matrix,
    and dimensions of the block */
  dim3 grid((dx+block.x-1)/block.x, (dy+block.y-1)/block.y); 

  printf("Kernel launch dimensions: \n");
  printf("\tGrid size  : {%d, %d, %d} blocks.\n",grid.x, grid.y, grid.z);
  printf("\tBlock size : {%d, %d, %d} threads.\n",block.x, block.y, block.z);

  // Calcuate A+B=C on the GPU
  add_matrices<<<grid, block>>>(d_A, d_B, d_C, dx, dy);
  //cudaCheckErrors("kernel launch failure");

  // Block the CPU until GPU finishes execution
  cudaDeviceSynchronize();

  // Copy the result matrix C from device to host
  cudaMemcpy(h_C, d_C, dx*dy*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("Kernel execution failure or cudaMemcpy H2D failure");

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaCheckErrors("cudaFree failure");

  // Reset Device
  cudaDeviceReset();
}
