#include <openacc.h>
#include "pch.h"
#include <stdio.h>


// Calculate A+B=C on the device
void openacc_matrix_add(const float *o_A, const float *o_B, float *o_C, const int rows, \
const int cols) {
  #pragma acc data copyin(o_A[0:rows*cols], o_B[0:rows*cols]) copyout(o_C[0:rows*cols]) 
  {
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++) {
        int idx = i*cols+j;
        o_C[idx] = o_A[idx] + o_B[idx];
      }
    }
  }
}
