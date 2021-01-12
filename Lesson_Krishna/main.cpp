#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "pch.h"

int main(int argc, char* argv[]) {
  float *h_A, *h_B, *h_C, *h_check;
  clock_t t0, t1, t2;
  double t1sum = 0.0;
  double t2sum = 0.0;
  int dx, dy; 

  if (argc > 1 && argc < 4) {
    dx = atoi(argv[1]);
    dy = atoi(argv[2]);
  } else if (argc >= 4){
    printf("Usage: ./executable dimX dimY\n");
    return -1;
  } else {
    dx = DEFAULT_DIM;
    dy = DEFAULT_DIM;
  }

  t0 = clock();

  // Allocate host matrices
  h_A = (float*)malloc(dx*dy*sizeof(float));
  h_B = (float*)malloc(dx*dy*sizeof(float));
  h_C = (float*)malloc(dx*dy*sizeof(float));
  h_check = (float*)malloc(dx*dy*sizeof(float));
  
  // Init matrices
  InitializeMatrixSame(h_A, dx, dy, MATRIX_ADD_A_VAL);
  InitializeMatrixSame(h_B, dx, dy, MATRIX_ADD_B_VAL);

  // Calculate A+B=C on the host
  cpu_matrix_add(h_A, h_B, h_check, dx, dy);

  // Printout for debugging
  // PrintMatrix(h_check, m, q);

  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds. Begin compute.\n", t1sum);
  
  // Calcuate A+B=C on the device
  gpu_matrix_add(h_A, h_B, h_C, dx, dy);
  
  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf("Done. Compute took %f seconds\n", t2sum);

  // Check for correctness
  MatrixVerification(h_check, h_C, dx, dy, MATRIX_ADD_TOL);
  
  // Cleanup
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_check);
  return 0;
}

