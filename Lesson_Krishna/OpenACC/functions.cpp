#include "pch.h"

void cpu_matrix_add(const float *A, const float *B, float *C, const int dx, \
const int dy) {
  for (int i = 0; i < dx; i++){
    for (int j = 0; j < dy; j++) {
      int idx = i*dx+j;
      C[idx] = A[idx] + B[idx];
    }
  }
}


