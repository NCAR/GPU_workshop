#include <cuda_runtime.h>
#include <cuda.h>
#include "pch.h"
#include <stdio.h>
 
__global__ void add_matrices(float* m1, float* m2, float* sum, int ny, int nx){

    // Calculate the thread x and y index 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate thread id
    int index = idy*nx+idx; 
    if (idx < nx && idy < ny) sum[index] = m1[index]+m2[index];
}

__host__ void gpu_matrix_add(const float *h_A, const float *h_B, float *h_C,\
   const int ny, const int nx)
{
  float *d_A, *d_B, *d_C;

  // 3. Allocate memory on GPU to matrices 
  cudaMalloc(&d_A, ny*nx*sizeof(float));
  cudaMalloc(&d_B, ny*nx*sizeof(float));
  cudaMalloc(&d_C, ny*nx*sizeof(float));

  // Check for any errors
  cudaCheckErrors("cudaMalloc failure");

  // 4. Copy the matrices A, B from host to device
  cudaMemcpy(d_A, h_A, ny*nx*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, ny*nx*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failture");

  // Kernel Invoke Paramters (2D grid and blocks) 
  // Decide the dimensions of a block
  int dimx = 16; 
  int dimy = 16; 
  // Block of 256 threads 
  dim3 block(dimx, dimy); 

  /* Calculate the grid dimensions based on the dimensions of the matrix,
    and dimensions of the block */
  dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y); 

  printf("Kernel launch dimensions: \n");
  printf("\tGrid size  : {%d, %d, %d} blocks.\n",grid.x, grid.y, grid.z);
  printf("\tBlock size : {%d, %d, %d} threads.\n",block.x, block.y, block.z);

  // 6. Launch the kernel to perform A+B=C on GPU
  add_matrices<<<grid, block>>>(d_A, d_B, d_C, ny, nx);
  cudaCheckErrors("kernel launch failure");

  // Block the CPU until GPU finishes execution
  cudaDeviceSynchronize();

  // 6. Copy the result matrix C from device to host
  cudaMemcpy(h_C, d_C, ny*nx*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("Kernel execution failure or cudaMemcpy H2D failure");

  // 7. Free the memory on GPU
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaCheckErrors("cudaFree failure");

  // 8. Reset Device
  cudaDeviceReset();
}
