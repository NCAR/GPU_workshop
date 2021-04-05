/* common.cpp
 * To contain functions that are common across all lessons
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define RANGE_MAX 1.0 
#define RANGE_MIN -1.0

/* Sets all values in array equal to val */
void InitializeMatrixSame(float *array, const int ny, const int nx, const float val, const char* name){
	// p serves as another pointer to the start rows within array
	float *p = array;

	for(int i=0; i<ny; i++){
		for(int j=0; j<nx; j++){
			p[j] = val;
		}
		// Advance p to the next row
		p += nx;
	}
        printf("Initialized Matrix %s, %d X %d \n",name, ny, nx);
        
      
}

/* Sets all elements of array to a number between [RANGE_MIN,RANGE_MAX] */
void InitializeMatrixRand(float *array, const int ny, const int nx,const char* name){
	// p serves as another pointer to the start rows within array
	float *p = array;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			p[j] = ((float)rand() / (RAND_MAX)*(RANGE_MAX - RANGE_MIN) + RANGE_MIN);
		}
		// Advance p to the next row
		p += nx;
	}
        printf("Initialized Random Matrix %s, %d X %d \n",name, ny, nx);
}

/* Compares the matrices element-wise and prints an error message if 
 * the difference between values is above fTolerance
 */
void MatrixVerification(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance){
	// Pointers for rows in each matrix
	float *p = hostC;
	float *q = gpuC;
        float *err = (float*)malloc(ny*nx*sizeof(float));
        bool PassFlag = 1;
        float maxErr = 0.0;
        float tmpErr = 0.0;
        float avgErr = 0.0;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
                   
                   tmpErr = fabs(p[j] - q[j]);
                   err[j] = tmpErr;
                   avgErr += tmpErr;
                   if(tmpErr > maxErr) {
                      maxErr = tmpErr;
                   }

	           if (fabs(p[j] - q[j]) > fTolerance){
                      PassFlag=0;
	 	   }
		}
		p += nx;
		q += nx;
              err += nx;
	}
        if(PassFlag){
           printf("Verification passed\n");
        } else {
           printf("Verification failed\n");
        }
        printf("\nAverage Error: %f", avgErr/(nx*ny));
        printf("\nMax Error    : %f", maxErr);
                      
}

void PrintMatrix(float *matrix, int ny, int nx){
	if (ny <= 6 && nx <= 6)
	{
		float *p = matrix;

		for (int i = 0; i < ny; i++)
		{
			for (int j = 0; j < nx; j++)
			{
				printf("%f\t",p[j]);
			}
			printf("\n");
			fflush(stdout);
			p += nx;
		}
	}
}

void copyMatrix(float *src, float *dest, const int ny, const int nx){
	float *p = src;
	float *q = dest;
	for(int i=0; i<ny; i++){
		for(int j=0; j<nx; j++){
			q[j] = p[j];
		}
		p += nx;
		q += nx;
	}
}
