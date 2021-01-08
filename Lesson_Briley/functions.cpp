//Function File for FMA code 

#include "Wkshp_head.h" 

void InitializeM(float *array, const int ny, const int nx,const float val)
{
	float *p = array; 
	for (int i=0; i<ny; i++) 
	{
		for(int j=0; j<nx; j++)
		{	
			p[j] = val; 
		}
		p += nx; 
	}
} 


//Square Matrix Multiplication 
void cpuMMult(float *A, float *B, float *C, const int nx, const int ny) 
{
	float fSum; 
	for (int i=0; i<ny;i++)
	{
		for(int j = 0; j<nx; j++)
		{
			fSum = 0.0f; 
			for(int k=0; k<nx; k++) 
			{
				fSum +=(A[(i*nx)+k]*B[(k*nx)+j]);//linear adressing multiplication
			}
			C[(i*nx)+j] = fSum; 
		}
	}
}

