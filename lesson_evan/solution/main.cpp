#include <stdio.h>
#include <stdlib.h>
//#include <cuda_runtime.h>
//#include <cuda.h>
#include <time.h>
#include "pch.h"

int main(int argc, char* argv[]) {
  float *h_A, *h_B, *h_C, *h_check;
  clock_t t0, t1, t2;
  double t1sum = 0.0;
  double t2sum = 0.0;
  int m, n, p, q;

  if (argc > 1 && argc < 6) {
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    p = atoi(argv[3]);
    q = atoi(argv[4]);
  } else if (argc >= 6){
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

  t0 = clock();

  // Allocate host matrices
  h_A = (float*)malloc(m*n*sizeof(float));
  h_B = (float*)malloc(p*q*sizeof(float));
  h_C = (float*)malloc(m*q*sizeof(float));
  h_check = (float*)malloc(m*q*sizeof(float));
  
  // Init matrices
  InitializeMatrixSame(h_A, m, n, MATMUL_A_VAL);
  InitializeMatrixSame(h_B, p, q, MATMUL_B_VAL);

  // Calculate AxB=C on the host
  cpuMatmul(h_A, h_B, h_check, m, p, q);

  // Printout for debugging
  //PrintMatrix(h_check, m, q);

  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds. Begin compute.\n", t1sum);
  
  // Calcuate AxB=C on the device
  gpuMatmul(h_A, h_B, h_C, m, p, q);
  
  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf("Done. Compute took %f seconds\n", t2sum);

  // Check for correctness
  MatrixVerification(h_check, h_C, m, q, MATMUL_TOL);
  
  // Cleanup
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_check);
  return 0;
}
