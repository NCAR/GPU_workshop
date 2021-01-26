#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "pch.h"

int main(int argc, char* argv[]) {
  float *o_A, *o_B, *o_C, *o_check;
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
  o_A = (float*)malloc(dx*dy*sizeof(float));
  o_B = (float*)malloc(dx*dy*sizeof(float));
  o_C = (float*)malloc(dx*dy*sizeof(float));
  o_check = (float*)malloc(dx*dy*sizeof(float));
  
  // Init matrices
  InitializeMatrixSame(o_A, dx, dy, MATRIX_ADD_A_VAL,"o_A");
  InitializeMatrixSame(o_B, dx, dy, MATRIX_ADD_B_VAL,"o_A");
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds. Begin CPU compute.\n", t1sum);

  // Calculate A+B=C on the host
  t0 = clock();
  cpu_matrix_add(o_A, o_B, o_check, dx, dy);
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("CPU Matrix Addition took %f seconds. \n", t1sum);

  // Printout for debugging
  printf("CPU Matrix Addition Results: \n");
  PrintMatrix(o_check, dx, dy);

  // Calculate A+B=C on the device using OpenACC
  t0 = clock();
  openacc_matrix_add(o_A, o_B, o_C, dx, dy);
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("GPU Matrix Addition took %f seconds. \n", t1sum);

  //Printout for debugging
  printf("GPU Matrix Addition Results: \n");
  PrintMatrix(o_C, dx, dy);
  
  // Check for correctness
  MatrixVerification(o_check, o_C, dx, dy, MATRIX_ADD_TOL);
  
  // Cleanup
  free(o_A);
  free(o_B);
  free(o_C);
  free(o_check);
  return 0;
}

