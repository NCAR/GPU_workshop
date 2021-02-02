#include<openacc.h>
#include "functions.h"


void OpenAccMult(const float *A,const float *B, float *C, const int m, const int p, const int q)
{

  // Calculate AxB=C on the GPU using openACC parallelization.
  float temp = 0.0;
  #pragma acc data copyout(C[m:q]) copyin(A[m:p],B[p:q])
  {
  #pragma acc parallel loop collapse(2) reduction(+:temp)
  for (int i = 0; i < m; i++)
    for (int j = 0; j < q; j++)
    {
        temp = 0.0;
        for (int k = 0; k < p; k++)
        {
            temp += A[i*p+k] * B[k*q+j];
        }

        C[i*q+j] = temp;
   }
  }
}

