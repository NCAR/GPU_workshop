#include "pch.h"

void gpuMatmul(const float *restrict A, const float *restrict B, float *restrict C, const int m, const int p, const int q) {
  float temp = 0.0;
  #pragma acc data copyin(A[0:m*p], B[0:p*q]) copyout(C[0:m*q])
  {
  for (int t = 0; t < 3; t++) {
    #pragma acc parallel loop gang collapse(2)
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < q; j++) {
        temp = 0.0;
        #pragma acc loop vector reduction(+:temp)
        for (int k = 0; k < p; k++) {
          temp += A[i*p+k] * B[k*q+j];
        }
        C[i*q+j] = temp;
        }
      }
    #pragma acc compare(C[0:m*q])
    }
  }
}
