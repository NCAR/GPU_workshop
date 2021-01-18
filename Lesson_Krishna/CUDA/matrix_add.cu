#include <cuda_runtime.h>
#include <cuda.h>
#include "pch.h"
#include <stdio.h>
 
__global__ void add_matrices(float* m1, float* m2, float* sum, int m, int n){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int index = idx*m+idy; 
    if (idx < m && idy < n) sum[index] = m1[index]+m2[index];
}

__host__ void gpu_matrix_add(const float *h_A, const float *h_B, float *h_C,\
   const int dx, const int dy)
{
  float *d_A, *d_B, *d_C;

  // Allocate device matrices
  cudaMalloc(&d_A, dx*dy*sizeof(float));
  cudaMalloc(&d_B, dx*dy*sizeof(float));
  cudaMalloc(&d_C, dx*dy*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, dx*dy*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, dx*dy*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failture");

  //Kernel Invoke Paramters (2D grid and blocks) 
  int dimx = 16; 
  int dimy = 16; 

  dim3 block(dimx, dimy); //Block of 256 threads 
  dim3 grid((dx+block.x-1)/block.x, (dy+block.y-1)/block.y); //grid dimensions 

  // Calcuate A+B=C on the device
  add_matrices<<<grid, block>>>(d_A, d_B, d_C, dx, dy);
  cudaCheckErrors("kernel launch failure");
  cudaDeviceSynchronize();
  cudaMemcpy(h_C, d_C, dx*dy*sizeof(float), cudaMemcpyDeviceToHost);

  cudaCheckErrors("Kernel execution failure or cudaMemcpy H2D failure");

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaCheckErrors("cudaFree failure");
  cudaDeviceReset();
}
