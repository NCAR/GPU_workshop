#include "pch.h"

int main(int argc, char* argv[]) {
  using namespace std::chrono;

  float *h_A, *h_B, *gpu_C, *cpu_C;
  high_resolution_clock::time_point t0, t1;
  duration<double>time_sum;
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

  t0 = high_resolution_clock::now();

  // Allocate host matrices
  h_A = (float*)malloc(m*n*sizeof(float));
  h_B = (float*)malloc(p*q*sizeof(float));
  gpu_C = (float*)malloc(m*q*sizeof(float));
  cpu_C = (float*)malloc(m*q*sizeof(float));
  
  // Initialize host matrices
  InitializeMatrixSame(h_A, m, n, MATMUL_A_VAL, "h_A");
  InitializeMatrixSame(h_B, p, q, MATMUL_B_VAL, "h_B");

  t1 = high_resolution_clock::now();
  time_sum = duration_cast<duration<double>>(t1-t0);
  printf("Init took %f seconds. Begin compute.\n", time_sum.count());
  
  t0 = high_resolution_clock::now(); 
  // Calculate AxB=C on the host
  cpuMatmul(h_A, h_B, cpu_C, m, p, q);
  t1 = high_resolution_clock::now();
  time_sum = duration_cast<duration<double>>(t1-t0);
  printf("CPU matrix multiplication took %f seconds\n", time_sum.count());
  
  t0 = high_resolution_clock::now();
  // Calcuate AxB=C on the device
  gpuMatmul(h_A, h_B, gpu_C, m, p, q);

  t1 = high_resolution_clock::now();
  time_sum = duration_cast<duration<double>>(t1-t0);
  printf("GPU matrix multiplication took %f seconds\n", time_sum.count());
  // Check for correctness
  MatrixVerification(cpu_C, gpu_C, m, q, MATMUL_TOL);

  // Free the host matrices
  free(h_A);
  free(h_B);
  free(gpu_C);
  free(cpu_C);
  return 0;
}
