#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "pch.h"
//#include "functions.h"
#include <ctime>
#include <chrono>
#include <ratio> 

int main(int argc, char* argv[]) {
  using namespace std::chrono;

  float *h_A, *h_B, *cpu_C, *gpu_C;
  high_resolution_clock::time_point t0, t1;
  duration<double> t1sum;
  int m, n, p, q;

  if (argc > 1 && argc < 6) {
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    p = atoi(argv[3]);
    q = atoi(argv[4]);
  } else if (argc >= 6){
    printf("Multiplies Matrices AxB=C");
    printf("Matrix A Dim: MxN");
    printf("Matrix B Dim: PxQ");
    printf("Matrix C Dim: MxQ");
    printf("Usage: ./executable M N P Q\n");
    return -1;
  } else {
    m = DEFAULT_DIM;
    n = DEFAULT_DIM;
    p = DEFAULT_DIM;
    q = DEFAULT_DIM;
  }
  if (n != p) {
    printf("Dimension N (%d) != dimension P (%d). Operation undefined.\n",
                    n, p);
    return -1;
  }

  t0 = high_resolution_clock::now();

  // Allocate memory to host matrices
  h_A = (float*)malloc(m*n*sizeof(float));
  h_B = (float*)malloc(p*q*sizeof(float));
  cpu_C = (float*)malloc(m*q*sizeof(float));
  gpu_C = (float*)malloc(m*q*sizeof(float));
  
  // Init matrices with default values of 3.0 for matrix A and 2.0 for matrix B
  InitializeMatrixSame(h_A, m, n, MAT_A_VAL,"h_A");
  InitializeMatrixSame(h_B, p, q, MAT_B_VAL,"h_B");

  t1 = high_resolution_clock::now();
  t1sum = duration_cast<duration<double>>(t1-t0);
  printf("Init took %f seconds. Begin compute.\n", t1sum.count());

  // Calculate A*B=C on the host
  t0 = high_resolution_clock::now();
  cpuMatmul(h_A,h_B,cpu_C,m,p,q);
  t1 = high_resolution_clock::now();
  t1sum = duration_cast<duration<double>>(t1-t0);
  printf("CPU Matrix Multiplication took %f seconds.\n", t1sum.count());

  // Calcuate A*B=C on the device
  t0 = high_resolution_clock::now();
  gpuMult(h_A,h_B,gpu_C,m,n,p,q,BLOCK_SIZE);
  t1 = high_resolution_clock::now();
  t1sum = duration_cast<duration<double>>(t1-t0);
  printf("GPU Matrix Multiplication took %f seconds.\n", t1sum.count());

  // Printout for debugging
   if (m <= 6 && q <= 6) {
        printf("\nCPU Matrix Multiplication Results: \n");
        PrintMatrix(cpu_C, m, q);
        printf("\nGPU Matrix Multiplication Results: \n");
        PrintMatrix(gpu_C, m, q);
  } 
  // Check for correctness
  MatrixVerification(gpu_C, cpu_C, m, q, VERIF_TOL);
  
  // Cleanup
  free(h_A);
  free(h_B);
  free(cpu_C);
  free(gpu_C);
  return 0;
}

