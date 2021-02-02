#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <chrono>
#include <ratio> 
#include "pch.h"

int main(int argc, char* argv[]) {
  using namespace std::chrono;

  float *h_A, *h_B, *cpu_C, *gpu_C;
  high_resolution_clock::time_point t0, t1;
  duration<double> t1sum;
  int dx, dy; 

  // Parse command line arguments
  if (argc > 1 && argc < 4) {
    dx = atoi(argv[1]);
    dy = atoi(argv[2]);
  } else if (argc >= 4){
    printf("Usage: ./executable dimX dimY\n");
    return -1;
  } else {
    // set to default dimensions (1024) if no arguments are passed
    dx = DEFAULT_DIM;
    dy = DEFAULT_DIM;
  }

  t0 = high_resolution_clock::now();

  // 1. Allocate memory to host matrices
  h_A = (float*)malloc(dy*dx*sizeof(float));
  h_B = (float*)malloc(dy*dx*sizeof(float));
  cpu_C = (float*)malloc(dy*dx*sizeof(float));
  gpu_C = (float*)malloc(dy*dx*sizeof(float));
  
  // 2. Init matrices with default values of 3.0 for matrix A and 2.0 for matrix B
  InitializeMatrixSame(h_A, dy, dx, MAT_A_VAL,"h_A");
  InitializeMatrixSame(h_B, dy, dx, MAT_B_VAL,"h_B");

  t1 = high_resolution_clock::now();
  t1sum = duration_cast<duration<double>>(t1-t0);
  printf("Init took %f seconds. Begin compute.\n", t1sum.count());

  // Calculate A+B=C on the host
  t0 = high_resolution_clock::now();
  cpu_matrix_add(h_A, h_B, cpu_C, dy, dx);
  t1 = high_resolution_clock::now();
  t1sum = duration_cast<duration<double>>(t1-t0);
  printf("CPU Matrix Addition took %f seconds.\n", t1sum.count());

  // Calcuate A+B=C on the device
  t0 = high_resolution_clock::now();
  gpu_matrix_add(h_A, h_B, gpu_C, dy, dx);
  t1 = high_resolution_clock::now();
  t1sum = duration_cast<duration<double>>(t1-t0);
  printf("GPU Matrix Addition took %f seconds.\n", t1sum.count());

  // Printout for debugging
   if (dx <= 6 && dy <= 6) {
        printf("\nCPU Matrix Addition Results: \n");
        PrintMatrix(cpu_C, dy, dx);
        printf("\nGPU Matrix Addition Results: \n");
        PrintMatrix(gpu_C, dy, dx);
  } 

  // Check for correctness
  MatrixVerification(cpu_C, gpu_C, dy, dx, VERIF_TOL);
  
  // 7. Free host memory
  free(h_A);
  free(h_B);
  free(cpu_C);
  free(gpu_C);
  return 0;
}

