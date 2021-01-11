#include <cuda_runtime.h>
#include <cuda.h>
#include "pch.h"
#include <stdio.h>

#define TILE_WIDTH 16

__global__ void SharedMatmul(const float *A, const float *B, float *C, const int ny, const int nx)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Allocate memory on the shared memory to store elements of A and B
  __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

  // Compute global row and column indices
  unsigned int col = tx + (blockDim.x*bx);
  unsigned int row = ty + (blockDim.y*by);

  float temp = 0.0f;
  for (int tw_idx = 0; tw_idx < (nx / TILE_WIDTH); tw_idx++) {
    // Load global elements into the tiles on the shared memory
    // Each thread loads one element per loop
    s_A[ty][tx] = A[(row*nx) + (tw_idx*TILE_WIDTH) + tx]; // Read A row-wise
    s_B[ty][tx] = B[col + nx * (tw_idx*TILE_WIDTH + ty)]; // Read B col-wise
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++) {
      temp += s_A[ty][k] * s_B[k][tx];
    }
    __syncthreads();
  }
  C[row*nx + col] = temp;
}

__host__ void gpuMatmul(const float *h_A, const float *h_B, float *h_C, const int m, const int p, const int q)
{
  float *d_A, *d_B, *d_C;

  // Allocate device matrices
  cudaMalloc(&d_A, m*p*sizeof(float));
  cudaMalloc(&d_B, p*q*sizeof(float));
  cudaMalloc(&d_C, m*q*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, m*p*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, p*q*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failture");

  int x_blocks = p / BLOCK_SIZE;
  int y_blocks = m / BLOCK_SIZE;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(x_blocks, y_blocks);

  // Calcuate AxB=C on the device
  SharedMatmul<<<grid, block>>>(d_A, d_B, d_C, m, q);
  cudaCheckErrors("kernel launch failure");
  cudaDeviceSynchronize();
  cudaMemcpy(h_C, d_C, m*q*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("Kernel execution failure or cudaMemcpy H2D failure");

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaCheckErrors("cudaFree failure");
  cudaDeviceReset();
}
