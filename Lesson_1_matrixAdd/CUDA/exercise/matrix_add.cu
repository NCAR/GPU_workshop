#include <cuda_runtime.h>
#include <cuda.h>
#include "pch.h"
#include <stdio.h>
 
// * The funtion qualifier "__global__" specifies this function is
// * callable from the host and executed on the GPU
__global__ void add_matrices(float* m1, float* m2, float* sum, int ny, int nx){

    // * Calculate the thread idx and idy index using threadIdx, blockDim, and blockIdx.
    // * Uncomment the following lines and calculate the thread's index w.r.t. the grid:
    // int idx = ?;
    // int idy = ?;

    // * Calculate the global linear address:
    // int gla_index = ?;

    // * Calculate the matrix sum if GLA index is within the matrix bounds:
    // if ( ? < ? && ? < ?) ?;

}

// The funtion qualifier "__host__" specifies this function is
// executed on the CPU
__host__ void gpu_matrix_add(const float *h_A, const float *h_B, float *h_C,\
   const int ny, const int nx)
{
  float *d_A, *d_B, *d_C;

  // 3. * Allocate memory on GPU for the matrices.
  //    * Uncomment the following 3 lines and replace question marks with correct variables 
  //    * and size in bytes.
  // cudaMalloc(? , ?*?*sizeof(float));
  // cudaMalloc(? , ?*?*sizeof(float));
  // cudaMalloc(? , ?*?*sizeof(float));

  // Check for any errors
  cudaCheckErrors("cudaMalloc failure");

  // 4. * Copy the matrices A, B from host to device.
  //    * Uncomment the following 3 lines and fill in the correct destination, source, 
  //    * and size in bytes:
  // cudaMemcpy(?, ?, ?, cudaMemcpyHostToDevice);
  // cudaMemcpy(?, ?, ?, cudaMemcpyHostToDevice);
  // cudaMemcpy(?, ?, ?, cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failture");

  // * Kernel Invoke Paramters (Grid & Blocks are 3D, but this is a 2D problem) 
  // * Any dimension not assigned is automatically set to 1
  // * Decide the dimensions of a block:
  int dimx = 16; 
  int dimy = 16; 

  // * To create a block of dimx*dimy threads,
  // * uncomment the following line and fill in the block dimensions: 
  // dim3 block(?, ?); 

  // * Based on the number of threads in each block, dynamically calculate the minimum
  // * number of blocks in the x and y dimensions for any size input matrix.
  // * Uncomment the following line and fill in the calculation for blocks in each dimension:
  // dim3 grid( ?, ?); 

  printf("Kernel launch dimensions: \n");
  printf("\tGrid size  : {%d, %d, %d} blocks.\n",grid.x, grid.y, grid.z);
  printf("\tBlock size : {%d, %d, %d} threads.\n",block.x, block.y, block.z);
  
  // 5. * Launch the kernel to perform A+B=C on the GPU.
  //    * Uncomment the following line and fill in the gridDim, blockDim, and input variables:
  // add_matrices<<<? , ?>>>(?, ?, ?, ?, ?);
  cudaCheckErrors("kernel launch failure");

  //    * Block the CPU until GPU finishes execution.
  cudaDeviceSynchronize();

  // 6. * Copy the result matrix C from device to host
  //    * Uncomment the following command and fill in correct destination, source, size in bytes,
  //    * and direction of copy:
  // cudaMemcpy(?, ?, ?, ?);

  cudaCheckErrors("Kernel execution failure or cudaMemcpy H2D failure");

  // 7. * Free the memory on GPU.
  //    * Uncomment and complete the following 3 lines to deallocate the device memory
  // cudaFree(?);
  // cudaFree(?);
  // cudaFree(?);
  cudaCheckErrors("cudaFree failure");

  // 8. * Reset Device
  cudaDeviceReset();
}
