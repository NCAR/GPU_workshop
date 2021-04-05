#include "pch.h"

// Calculate A+B=C on the host
void cpu_matrix_add(const float *A, const float *B, float *C, const int rows, \
const int cols) {
  for (int i = 0; i < rows; i++){
    for (int j = 0; j < cols; j++) {
      int idx = i*cols+j;
      C[idx] = A[idx] + B[idx];
    }
  }
}
