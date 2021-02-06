/* stencil.cc
 */

#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "pch.h"

#define Swap(x,y) {float* temp; temp = x; x = y; y = temp;}

void copyMatrix_ACC(float *src, float *dest, const int ny, const int nx){
	printf("In copyMatrix_ACC\n");
	// #pragma acc data copyin(src[0:ny*nx]) copyout(dest[0:ny*nx])
	// {
	// #pragma acc parallel loop collapse(2)
        for(int i=0; i<ny; i++){
                for(int j=0; j<nx; j++){
                        dest[i*nx+j] = src[i*nx+j];
                }
        }
	// } // acc end data
	printf("Exiting copyMatrix_ACC\n");
}


float Jacobi_ErrorCalcACC(const float *A, const float *b, const float *x, const int ny, const int nx){
	// Perform Ax-b and return the sum of the entires
        float sum = 0.0;
        float error = 0.0; // L1 norm of Ax-b
	// #pragma acc data copyin(A[0:ny*nx], b[0:ny], x[0:ny], sum, error) copyout(error)
	// {
	// #pragma acc parallel loop gang reduction(+:sum)
        for(int i=0; i<ny; i++){
		// #pragma acc loop vector reduction(+:sum)
                for(int j=0; j<nx; j++){
                        sum += A[i*nx+j]*x[j];
                }
                error += fabs(sum - b[i]);
                sum = 0.0;
        }
	// } // acc end data
        return error;
}


void Jacobi_naiveACC(const float *A, const float *b, float *x, const int ny, const int nx, const float threshold){
	int itr = 0;
	float error = FLT_MAX;
	float sum1 = 0.0f, sum2=0.0f;
	float *x_new;
	x_new = (float*)malloc(ny*sizeof(float));

	while(itr < JACOBI_MAX_ITR and error > threshold){
	// #pragma acc data copyin(A[0:ny*nx], b[0:ny], x[0:ny], itr) \
	// copyout(x[0:ny], error, itr) \
	// create(x_new[0:ny], sum1, sum2)
	// {
		// #pragma acc parallel loop gang private(b)	
		for(int i=0; i<ny; i++){
			// #pragma acc loop vector private(A, sum1)
			for(int j=0; j<i; j++){
				sum1 += A[i*nx+j]*x[j];
			}
			// #pragma acc loop vector private(A, sum2)
			for(int j=i+1; j<nx; j++){
				sum2 += A[i*nx+j]*x[j];
			}
			// Find the new value of x for this entry
			x_new[i] = 1/A[i*nx+i] *(b[i] - sum1 - sum2);
			// Reset sum
			sum1 = sum2 = 0.0;
		}
		// Copy x_new into x
		Swap(x,x_new);
		itr += 1;
		// Check the error
		error = Jacobi_ErrorCalcACC(A, b, x, nx, ny);
	// } // acc end data
	}
	printf("ACC Jacobi exiting on itr %d with error %f\n", itr, error);
	free(x_new);
}
