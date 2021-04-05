#include <openacc.h>
#include "pch.h"
#include <stdio.h>


void openacc_matrix_mult(const float *A,const float *B, float *C, const int rowsA, const int colsB, const int rowsB)
{
  
  float temp = 0.0;
  #pragma acc data copyout(C[0:rowsA*colsB]) copyin(A[0:rowsA*rowsB],B[0:rowsB*colsB])
  {
  #pragma acc parallel loop collapse(2) reduction(+:temp)
  for (int i = 0; i < rowsA; i++)
    for (int j = 0; j < colsB; j++)
    {
        temp = 0.0;
   #pragma acc loop vector reduction(+:temp)
        for (int k = 0; k < rowsB; k++)
        {
            temp += A[i*rowsB+k] * B[k*colsB+j];
        }

        C[i*colsB+j] = temp;
   }
  }
}


