#include "pch.h"

// Calculate A+B=C on the host
void cpu_matrix_add(const float *A, const float *B, float *C, const int dx, \
const int dy) {
  for (int i = 0; i < dx; i++){
    for (int j = 0; j < dy; j++) {
      int idx = i*dy+j;
      C[idx] = A[idx] + B[idx];
    }
  }
}


