/* functions.cpp
 * Contains the host-side functions specific to this problem
 */

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "pch.h"

void InitializeDiagDomMat(float *array, const int ny, const int nx, const char *name){
	float *p;
	float rand_val = 0.0;
	float sum = 0.0;
	// First initialize the matrix with random numbers
	InitializeMatrixRand(array, ny, nx, name);
        // Iterate through the diagonal elements and multiply
        // each by a random float to make it greater than the
        // other elements
        p = array;

        for(int i = 0; i < ny; i++){
		for(int j = 0; j<ny; j++){
			sum += fabs(p[j]);
		}
		rand_val = ((float)rand() / (RAND_MAX))*(DIAG_MAX-DIAG_MIN)+DIAG_MIN;
		p[i] = rand_val*sum;
		p += nx;
	}
	
}

bool CheckDiagDomMat(float *array, const int ny, const int nx){
	float *p = array;
	float sum = 0.0;
	for(int i=0; i<ny; i++){
		for(int j=0; j<nx; j++){
			if(i != j){
				sum += fabs(p[j]);
			}
		}
		if(fabs(p[i]) < sum){
			return false;
		}
		p += nx;
		sum = 0.0;
	}
	return true;
}

float Jacobi_ErrorCalcCPU(const float *A, const float *b, const float *x, const int ny, const int nx){
	// Perform Ax-b and return the sum of the entries
	float sum = 0.0;
	float error = 0.0; // L1 norm of Ax-b
	for(int i=0; i<ny; i++){
		for(int j=0; j<nx; j++){
			sum += A[i*nx+j]*x[j];
		}
		error += fabs(sum - b[i]);
		sum = 0.0;
	}
	return error;

}

void Jacobi_naiveCPU(const float *A, const float *b, float *x, const int ny, const int nx, const float threshold){
	int itr = 0;
	float error = FLT_MAX;
	float sum = 0.0;
	float *x_new = new float[ny];

	while(itr < JACOBI_MAX_ITR and error > threshold){
		for(int i=0; i<ny; i++){
			for(int j=0; j<nx; j++){
				if(j != i){
					sum += A[i*nx+j]*x[j];
				}
			}
			// Find the new value of x for this entry
			x_new[i] = 1/A[i*nx+i] *(b[i] - sum);
			// Reset sum
			sum = 0.0;
		}
		// Copy x_new into x
		copyMatrix(x_new, x, 1, nx);
		itr += 1;
		// Check the error
		error = Jacobi_ErrorCalcCPU(A, b, x, nx, ny);
	}
	printf("Jacobi exiting on itr %d with error %f\n", itr, error);
	delete[] x_new;
}
