#include "pch.h"

void cpuMatmul(const float *A, const float *B, float *C, const int m, const int p, const int q) {
  // Calculate AxB=C on the host
  float temp = 0.0;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < q; j++) {
      temp = 0.0;
      for (int k = 0; k < p; k++) {
        temp += A[i*p+k] * B[k*q+j];
      }
      C[i*q+j] = temp;
      }
}
