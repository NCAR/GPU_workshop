#include <cuda_runtime.h>
#include <cuda.h>
#include "pch.h"

__global__ void SharedMatmul(const float *a, const float *b, float *c, const int m, const int p, const int q) {
  // Compute each thread's global row and column index.
  // int row = ???
  // int col = ???

  // Statically allocate a tile of shared memory. Tile size should equal the
  // number of threads per block.
  // ??? float s_a[???];
  // ??? float s_b[???];

  // Declare a temporary variable to accumulate calculated elements
  // for the C matrix.
  float tmp = 0.0;

  // Sweep tiles of size blockDim.x across matrices A and B.
  for (int i = 0; i < p; i += blockDim.x) {
   
    // Load in elements from A and B into shared memory into each tile.
    // int shared_index = ???

    // For matrix A, keep the row invariant and iterate through columns.
    // s_a[shared_index] = a[row * ??? + ??? + ???];

    // For matrix B, keep the column invariant and iterate through rows.
    // s_b[shared_index] = b[??? * ??? + ??? * ??? + col];

    // Wait for tiles to be loaded in before doing computation.

    // Do matrix multiplication on the small matrix within the current tile.
    // for (int j = 0; j < ???; j++) {
    //   tmp += s_a[??? * ??? + j] * s_b[j * ??? + ???];
    // }
 
    // Wait for all threads to finish using current tiles before loading in new ones.
  }

  // Write resulting calculations as elements of the C matrix.
  // c[row * q + col] = tmp;
}

__host__ void gpuMatmul(const float *h_A, const float *h_B, float *gpu_C, const int m, const int p, const int q)
{
  float *d_A, *d_B, *d_C;

  // Allocate device matrices on GPU using cudaMalloc
  cudaMalloc(&d_A, m*p*sizeof(float));
  cudaMalloc(&d_B, p*q*sizeof(float));
  cudaMalloc(&d_C, m*q*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  // Copy host matrices A and B to the device using cudaMemcpy
  cudaMemcpy(d_A, h_A, m*p*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, p*q*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failture");

  // Set threads per CUDA block dimension. The maximum number
  // of total threads is 1024.
  int thread_dim = BLOCK_SIZE;
  // Set blocks per grid dimension (assume thread_dim divides M and N evenly)
  int blocks_x = q / thread_dim;
  int blocks_y = m / thread_dim;
  // Use dim3 structs for block and grid dimensions
  dim3 threads(thread_dim, thread_dim);
  dim3 blocks(blocks_x, blocks_y);

  // Launch the kernel to calculate AxB=C on the device
  SharedMatmul<<<blocks, threads>>>(d_A, d_B, d_C, m, p, q);
  cudaCheckErrors("kernel launch failure");
  // Synchronize the device, then copy device's C matrix to the host
  cudaDeviceSynchronize();
  cudaMemcpy(gpu_C, d_C, m*q*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("Kernel execution failure or cudaMemcpy H2D failure");

  // Free the device matrices
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaCheckErrors("cudaFree failure");
}
