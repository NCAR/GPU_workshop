#include "functions.h"

__global__ void mmul( float *a, float *b, float *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m)
    {
        for(int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}



__host__ void gpuMult(float *h_A, float *h_B, float *h_check, const int m, const int n, const int p, const int q, const int block_size)
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
  
  //calculate grid and block dimensions
  unsigned int grid_rows = (m + block_size - 1) / block_size;
  unsigned int grid_cols = (q + block_size - 1) / block_size;
  dim3 grid(grid_cols, grid_rows);
  dim3 block(block_size, block_size);

  //carry out matrix multiplication on the GPUs
  mmul<<<grid,block>>>(d_A,d_B,d_C,m,n,q);
  cudaDeviceSynchronize();
  cudaCheckErrors("Kernel execution failure or cudaMemcpy H2D failure");
  

  // Transefr results from device to host 
  cudaMemcpy(h_check, d_C, sizeof(int)*m*q, cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaCheckErrors("cudaFree failure");
}
