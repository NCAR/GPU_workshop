#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "pch.h"

int main(int argc, char* argv[]) {
  float *h_A, *h_B, *h_C, *h_check;
  clock_t t0, t1;
  double t1sum = 0.0;
  int dx, dy; 

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

  t0 = clock();

  // Allocate memory to host matrices
  h_A = (float*)malloc(dx*dy*sizeof(float));
  h_B = (float*)malloc(dx*dy*sizeof(float));
  h_C = (float*)malloc(dx*dy*sizeof(float));
  h_check = (float*)malloc(dx*dy*sizeof(float));
  
  // Init matrices with default values of 3.0 for matrix A and 2.0 for matrix B
  InitializeMatrixSame(h_A, dx, dy, MATRIX_ADD_A_VAL,"h_A");
  InitializeMatrixSame(h_B, dx, dy, MATRIX_ADD_B_VAL,"h_B");

  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds. Begin compute.\n", t1sum);

  // Calculate A+B=C on the host
  t0 = clock();
  cpu_matrix_add(h_A, h_B, h_check, dx, dy);
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("CPU Matrix Addition took %f seconds.\n", t1sum);

  // Printout for debugging
  printf("CPU results: \n");
  PrintMatrix(h_check, dx, dy);

  // Calcuate A+B=C on the device
  t0 = clock();
  gpu_matrix_add(h_A, h_B, h_C, dx, dy);
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("GPU Matrix Addition took %f seconds.\n", t1sum);

  // Printout for debugging
   printf("GPU results: \n");
   PrintMatrix(h_C, dx, dy);
  
  // Check for correctness
  MatrixVerification(h_check, h_C, dx, dy, MATRIX_ADD_TOL);
  
  // Cleanup
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_check);
  return 0;
}

