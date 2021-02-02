//Function File for FMA code 

#include "Wkshp_head.h" 
#include <openacc.h>

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

//OpenAcc FMA execution
void ACC_FMA(float *A, float *B, float *C, float *D, const int nx, const int ny)
{
	float fSum; 
        //Initial transfer of data to the GPU
 	#pragma acc enter data copyin(fSum) create(D[:nx*ny])
	//Inform compiler that Matrix data has already been moved to the GPU
	#pragma acc data present(A[:nx*ny],B[:nx*ny],C[:nx*ny]) 	
	//Directive to indicate what should be executed on the GPU
	#pragma acc parallel loop  
	for (int i=0; i<ny;i++)
        {
		for(int j = 0; j<nx; j++)
                {
                        fSum = 0.0f;
			for(int k=0; k<nx; k++)
                        {
				fSum +=(A[(i*nx)+k]*B[(k*nx)+j]);
			}
			D[(i*nx)+j] = fSum + C[(i*nx)+j];
		}
	}
}
