#include "pch.h"

void gpuMatmul(const float *A, const float *B, float *C, const int m, const int p, const int q) {
  float temp = 0.0;
#pragma acc data copyin(A, B) copyout(C)
#pragma acc parallel loop tile(64,8) reduction(+:temp)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < q; j++) {
      temp = 0.0;
#pragma acc loop reduction(+:temp)
      for (int k = 0; k < p; k++) {
        temp += A[i*p+k] * B[k*q+j];
      }
      C[i*q+j] = temp;
      }
  }
}
