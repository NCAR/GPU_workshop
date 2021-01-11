//Function File for FMA code 

#include "Wkshp_head.h" 


//FMA operation 
void CPU_FMA(float *A, float *B, float *C, float *D, const int nx, const int ny)
{
        float fSum;
        for (int i=0; i<ny;i++)
        {
                for(int j = 0; j<nx; j++)
                {
                        fSum = 0.0f;
                        for(int k=0; k<nx; k++)
                        {
				//linear adressing multiplication 
                                fSum +=(A[(i*nx)+k]*B[(k*nx)+j]);
                        }
			//Addition
                        D[(i*nx)+j] = fSum + C[(i*nx)+j];
                }
        }
}

