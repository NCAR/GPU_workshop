#include "pch.h"

// Calculate A*B=C on the host
void cpu_matrix_mult(const float *A, const float *B, float *C, const int rowsA, const int colsB, const int rowsB) 
{

 float temp = 0.0;
  for (int i = 0; i < rowsA; i++)
      for (int j = 0; j < colsB; j++)
      {
           temp = 0.0;
           for (int k = 0; k < rowsB; k++)
           {
               temp += A[i*rowsB+k] * B[k*colsB+j];
           }
           C[i*colsB+j] = temp;
      }
}


