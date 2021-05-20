#include <openacc.h>
#include "pch.h"
#include <stdio.h>


void openacc_matrix_mult(const float *A,const float *B, float *C, const int rowsA, const int colsB, const int rowsB)
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

void openacc_matrix_mult_trans(const float *A,const float *B, float *C, const int rowsA, const int colsB, const int rowsB)
{
float temp = 0.0;
 float *t_B;

 t_B = (float*)malloc(rowsB*colsB*sizeof(float));

#pragma acc data copyout(C[0:rowsA*colsB]) \
     copyin(A[0:rowsA*rowsB],B[0:rowsB*colsB]) \
     create(t_B[0:rowsB*colsB])
  {

#pragma acc parallel loop collapse(2) present(B[0:rowsB*colsB],t_B[0:rowsB*colsB])
 for (int i = 0; i < rowsB; i++)
      for (int j = 0; j < colsB; j++)
      {
         t_B[j*rowsB+i] = B[i*colsB+j];

      }
 #pragma acc parallel loop collapse(2) \
     reduction(+:temp)
  for (int i = 0; i < rowsA; i++)
      for (int j = 0; j < colsB; j++)
      {
           temp = 0.0;
  #pragma acc loop reduction(+:temp)
           for (int k = 0; k < rowsB; k++)
           {
               temp += A[i*rowsB+k] * t_B[i*rowsB+k];
           }
           C[i*colsB+j] = temp;
      }
  }
   free(t_B);
}

