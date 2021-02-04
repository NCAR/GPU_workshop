//Function File for FMA code 

#include "pch.h" 


//FMA operation 
void CPU_FMA(float *A, float *B, float *C, float *D, const int rows, const int cols)
{
        float fSum;
        for (int i=0; i<cols;i++)
        {
                for(int j = 0; j<rows; j++)
                {
                        fSum = 0.0f;
                        for(int k=0; k<rows; k++)
                        {
				//linear adressing multiplication 
                                fSum +=(A[(i*rows)+k]*B[(k*rows)+j]);
                        }
			//Addition
                        D[(i*rows)+j] = fSum + C[(i*rows)+j];
                }
        }
}

