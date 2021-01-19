#include "functions.h"
#include <stdio.h>
#include <stdlib.h>
//#include <cuda_runtime.h>
//#include <cuda.h>
#include <time.h>

const int default_dim = 1024;
const int block_size = 32; // The CUDA max is 1024 threads per block

int main(int argc, char* argv[]) {
  float *h_A, *h_B, *h_C, *h_check;
  clock_t t0, t1;
  double t1sum = 0.0;
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
    m = default_dim;
    n = default_dim;
    p = default_dim;
    q = default_dim;
  }
  if (n != p) {
    printf("Dimension N (%d) != dimension P (%d). Operation undefined.\n",
                    n, p);
    return -1;
  }


  //Start timer for CPU process
  t0 = clock();

  // Allocate memory for the host matrices
  h_A = (float*)malloc(m*n*sizeof(float));
  h_B = (float*)malloc(p*q*sizeof(float));
  h_C = (float*)malloc(m*q*sizeof(float));
 
  
  //initialize matrices on the host
  InitializeMatrixSame(h_A, m, n, MATMUL_A_VAL);
  InitializeMatrixSame(h_B, p, q, MATMUL_B_VAL);
  InitializeMatrixSame(h_C, m, q, MATMUL_B_VAL);
  

  //carry out cpu matrix multiplication
  cpuMatmul(h_A,h_B,h_C,m,p,q);

  //End timer for CPU Process
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Entire CPU process took %f seconds.... Starting OpenAcc Process.\n", t1sum);

  //Initialize check matrix to be used for verification
  h_check = (float*)malloc(m*q*sizeof(float));
  InitializeMatrixSame(h_check, m, q, MATMUL_B_VAL);


  //OpenAcc matrix multiplication
  //Start timer for CPU process
  t0 = clock();

  //carry out openacc matrix multiplication
  OpenAccMult(h_A,h_B,h_check,m,p,q);

  //stop timer
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Entire OpenAcc calculation took %f seconds....\n", t1sum);


  // Check for correctness. Need to agree on a tolerance value
  MatrixVerification(h_C, h_check, m,q,MATMUL_TOL);

  // Start timer for GPU process
  t0 = clock();

  //carry out GPU matrix multiplication
  gpuMult(h_A,h_B,h_check,m,n,p,q,block_size);
  
  //End timer for CPU process
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Done. Matrix multiplication on GPU took %f seconds.\n", t1sum);

  // Check for correctness. Need to agree on a tolerance value
  MatrixVerification(h_C, h_check, m,q,MATMUL_TOL);
  
  // Cleanup
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_check);          
  return 0;
}
