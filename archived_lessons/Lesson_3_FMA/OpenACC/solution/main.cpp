#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "pch.h"
#include <ctime>
#include <chrono>
#include <ratio> 

int main(int argc, char* argv[]) {
  using namespace std::chrono;

  float *h_A, *h_B,*cpu_C, *gpu_C;
  high_resolution_clock::time_point t0, t1;
  duration<double> t1sum;
  int rowsA, colsA, rowsB, colsB; 

  if (argc > 1 && argc < 6) {
    rowsA = atoi(argv[1]);
    colsA = atoi(argv[2]);
    rowsB = atoi(argv[3]);
    colsB = atoi(argv[4]);
  } else if (argc >= 6){
    printf("Usage: ./matrix_mult.exe rowsA colsA rowsB colsB\n");
    return -1;
  } else {
    rowsA = DEFAULT_DIM;
    colsA = DEFAULT_DIM;
    rowsB = DEFAULT_DIM;
    colsB = DEFAULT_DIM;
  }
  if (colsA != rowsB) {
    printf("Inner Dimension Mismatch... (%d) != (%d). Operation undefined.\n",
                    colsA, rowsB);
    return -1;
  }

  t0 = high_resolution_clock::now();

  // Allocate host matrices
  h_A = (float*)malloc(rowsA*colsA*sizeof(float));
  h_B = (float*)malloc(rowsB*colsB*sizeof(float));
  cpu_C = (float*)malloc(rowsA*colsB*sizeof(float));
  gpu_C = (float*)malloc(rowsA*colsB*sizeof(float));
  
  // Init matrices
  InitializeMatrixRand(h_A, rowsA, colsA,"h_A");
  InitializeMatrixRand(h_B, rowsB, colsB,"h_B");

  t1 = high_resolution_clock::now();
  t1sum = duration_cast<duration<double>>(t1-t0);
  printf("Init took %f seconds. Begin CPU compute.\n", t1sum.count());

  // Calculate A+B=C on the host
  t0 = high_resolution_clock::now();
  cpu_matrix_mult(h_A, h_B, cpu_C, rowsA, colsB, rowsB);
  t1 = high_resolution_clock::now();
  t1sum = duration_cast<duration<double>>(t1-t0);
  printf("CPU Matrix Multiplication & Addition took %f seconds. \n", t1sum.count());

  // Calculate A+B=C on the device using OpenACC
  t0 = high_resolution_clock::now();
  #pragma acc enter data copyin(h_A[:rowsA*colsA],h_B[:rowsB*colsB]) create(gpu_C[:rowsA*colsB])
  openacc_matrix_mult(h_A, h_B, gpu_C, rowsA, colsB, rowsB);
  #pragma acc exit data copyout(gpu_C[:rowsA*colsB]) delete(h_A[:rowsA*colsA],h_B[:rowsB*colsB])
  t1 = high_resolution_clock::now();
  t1sum = duration_cast<duration<double>>(t1-t0);
  printf("GPU Matrix Multiplication & Addition took %f seconds. \n", t1sum.count());

  // If matrices are small, print results
  if (rowsA <= 6 && colsB <= 6) {
  	printf("\nCPU Matrix Mult Results: \n");
 	PrintMatrix(cpu_C, rowsA, colsB);
 	printf("\nGPU Matrix Mult Results: \n");
  	PrintMatrix(gpu_C, rowsA, colsB);
  }
  // Check for correctness
  MatrixVerification(cpu_C, gpu_C, rowsA, colsB, VERIF_TOL);
  
  // Cleanup
  free(h_A);
  free(h_B);
  free(cpu_C);
  free(gpu_C);
  return 0;
}

