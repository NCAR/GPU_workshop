#include <openacc.h>
#include "pch.h"
#include <stdio.h>


// Calculate A+B=C on the device
void openacc_matrix_add(const float *o_A, const float *o_B, float *o_C, const int dx, \
const int dy) {
  #pragma acc data copyin(o_A[0:dx*dy], o_B[0:dx*dy]), copy(o_C[0:dx*dy]) 
  {
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < dx; i++){
      for (int j = 0; j < dy; j++) {
        int idx = i*dx+j;
        o_C[idx] = o_A[idx] + o_B[idx];
      }
    }
  }
}
