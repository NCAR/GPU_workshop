#include <cuda_runtime.h>
#include <cuda.h>
#include "pch.h"


__global__ void jacobi()
{

}

__host__ void Jacobi_naiveGPU(const float *A, const float *b, float *x, const int ny, const int nx, const float threshold)
{
  float *d_A, *d_b, *d_x;

  // Allocate device matrices on GPU using cudaMalloc
  cudaMalloc(&d_A, ny*nx*sizeof(float));
  cudaMalloc(&d_b, *q*sizeof(float));
  cudaMalloc(&d_x, m*q*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");


  // Free the device matrices
  cudaFree(d_A);
  cudaFree(d_b);
  cudaFree(d_x);
  cudaCheckErrors("cudaFree failure");

}
