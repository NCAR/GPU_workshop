#include "pch.h"

int main(int argc, char* argv[]) {
  float *h_A, *h_B, *gpu_C, *cpu_C;
  clock_t t0, t1, t2, t3, t4, t5;
  double t1sum = 0.0;
  double t3sum = 0.0;
  double t5sum = 0.0;
  int m, n, p, q;

  // Parse command line input (if any)
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
  gpu_C = (float*)malloc(m*q*sizeof(float));
  cpu_C = (float*)malloc(m*q*sizeof(float));
  
  // Initialize host matrices
  InitializeMatrixSame(h_A, m, n, MATMUL_A_VAL);
  InitializeMatrixSame(h_B, p, q, MATMUL_B_VAL);

  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds. Begin compute.\n", t1sum);
  
  t2 = clock(); 
  // Calculate AxB=C on the host
  cpuMatmul(h_A, h_B, cpu_C, m, p, q);
  t3 = clock();
  t3sum = ((double)(t3-t2))/CLOCKS_PER_SEC;
  printf("CPU done. Compute took %f seconds\n", t3sum);
  
  t4 = clock();
  // Calcuate AxB=C on the device with CUDA
  gpuMatmul(h_A, h_B, gpu_C, m, p, q);

  t5 = clock();
  t5sum = ((double)(t5-t4))/CLOCKS_PER_SEC;
  printf("CUDA done. Compute took %f seconds\n", t5sum);
  // Check for correctness
  MatrixVerification(cpu_C, gpu_C, m, q, MATMUL_TOL);

  // Free the host matrices
  free(h_A);
  free(h_B);
  free(gpu_C);
  free(cpu_C);
  return 0;
}
