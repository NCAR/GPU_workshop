#include "pch.h"

void gpuMatmul(const float *A, const float *B, float *C, const int m, const int p, const int q) {
  float temp = 0.0;
  #pragma acc data copyin(A[0:m*p], B[0:p*q]) copyout(C[0:m*q]) create(temp)
  {
  #pragma acc parallel loop gang vector collapse(2) reduction(+:temp)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < q; j++) {
      temp = 0.0;
      #pragma acc loop 
      for (int k = 0; k < p; k++) {
      #pragma acc cache(A[i*p:p], B[k*q:j])
      //#pragma acc cache(A[i*p:32])
        temp += A[i*p+k] * B[k*q+j];
      }
      C[i*q+j] = temp;
      }
  }
  }
}
