#include <openacc.h>
#include "pch.h"
#include <stdio.h>
 
void openacc_matrix_add(const float *A, const float *B, float *C, const int dx, \
const int dy) {
  #pragma acc data copyin(A[0:dx*dy], B[0:dx*dy]), copy(C[0:dx*dy]) 
  {
    #pragma acc kernels
    for (int i = 0; i < dx; i++){
      for (int j = 0; j < dy; j++) {
        int idx = i*dx+j;
        C[idx] = A[idx] + B[idx];
      }
    }
  }
}
