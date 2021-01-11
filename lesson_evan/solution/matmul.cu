#include <cuda_runtime.h>
#include <cuda.h>
#include "pch.h"
#include <stdio.h>

#define TILE_WIDTH 16

/*
// Code from Dr. Suresh via Dylan. Couldn't get it to work for
// rectangular matrices...
__global__ void SharedMatmul(float *g_A, float *g_B, float *g_C, const int ny, const int nx)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  //Allocate memory on the shared memory to store elements of A and B of the TILE_WIDTH x TILE_WIDTH size equal to a block
  __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

  //Compute global row and column indices
  unsigned int col = tx + (blockDim.x*bx);
  unsigned int row = ty + (blockDim.y*by);

  float fSum = 0.0f;
  for (int tw_idx = 0; tw_idx < (nx / TILE_WIDTH); tw_idx++) {
  // Load global elements into the tiles on the shared memory, each thread loads one element per loop
    s_A[ty][tx] = g_A[(row*nx) + (tw_idx*TILE_WIDTH) + tx]; // Read g_A row-wise
    s_B[ty][tx] = g_B[col + nx * (tw_idx*TILE_WIDTH + ty)]; // Read g_B column-wise
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++) {
      fSum += s_A[ty][k] * s_B[k][tx];
    }
    __syncthreads();
  }
  g_C[row*nx + col] = fSum;
}
*/

__global__ void SharedMatmul(const float *a, const float *b, float *c, const int M, const int K, const int N) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Statically allocated shared memory
  __shared__ float s_a[1024];
  __shared__ float s_b[1024];

  // Accumulate in temporary variable
  float tmp = 0.0;

  // Sweep tile across matrix
  for (int i = 0; i < K; i += blockDim.x) {
    // Load in elements for this tile
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] =
        b[i * N + threadIdx.y * N + col];

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix
    for (int j = 0; j < blockDim.x; j++) {
      tmp +=
          s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }

  // Write back results
  c[row * N + col] = tmp;
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

  /*
  // Dims that I had originally
  int x_blocks = p / BLOCK_SIZE;
  int y_blocks = m / BLOCK_SIZE;
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(x_blocks, y_blocks);
  */

  /* 
  // Dims adapted from Dylan's code
  int dimx = TILE_WIDTH;
  int dimy = TILE_WIDTH;
  dim3 block(dimx, dimy);
  dim3 grid((q + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  */

  // Dims adapted from Coffee's code
  // Threads per CTA dimension
  int THREADS = 32;
  // Blocks per grid dimension (assumes THREADS divides M and N evenly)
  int BLOCKS_X = q / THREADS;
  int BLOCKS_Y = m / THREADS;
  // Use dim3 structs for block and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS_X, BLOCKS_Y);

  // Calcuate AxB=C on the device
  SharedMatmul<<<blocks, threads>>>(d_A, d_B, d_C, m, p, q);
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
