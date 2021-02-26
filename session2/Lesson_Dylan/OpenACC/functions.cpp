/* functions.cpp
 * Contains the host-side functions specific to this problem
 * by: G. Dylan Dickerson (gdicker@ucar.edu)
 */

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "pch.h"

#define MAX(X,Y) (((X) > (Y)) ? (X) : (Y))

void LaplaceJacobi_naiveCPU(float *M, const int b, const int ny, const int nx, int& itr, float& error){
	// Use an iterative Jacobi solver to find the steady-state of the differential equation
	// of the Laplace eqution in 2 dimensions. M models the initial state of the system and
	// is used to return the result in-place. M has a border of b entries that aren't updated
	// by the Jacobi solver. For the iterative Jacobi method, the unknowns are a flattened
	// version of the interior points. See another source for more information
	//
	// The result is solving a system of the form 
	// 	M[i][j] = 1/4(M[i-1][j] + M[i][j+1] + M[i+1][j] + M[i][j-1])
	itr = 0;
	float maxdiff = 0.0f;
	float *M_new;

	M_new = (float*)malloc(ny*nx*sizeof(float));

	do {
		maxdiff = 0.0f;
		itr++;
		// Update M_new with M
		for(int i=b; i<ny-b; i++){
			for(int j=b; j<nx-b; j++){
				M_new[i*nx+j] = 0.25f *(M[(i-1)*nx+j]+M[i*nx+j+1]+ \
							M[(i+1)*nx+j]+M[i*nx+j-1]);
			}
		}

		// Check for convergence while copying values into M
		for(int i=b; i<ny-b; i++){
			for(int j=b; j<nx-b; j++){
				maxdiff = MAX(fabs(M_new[i*nx+j] - M[i*nx+j]), maxdiff);
				M[i*nx+j] = M_new[i*nx+j];
			}
		}
	} while(itr < JACOBI_MAX_ITR && maxdiff > JACOBI_TOLERANCE);
	printf("CPU Jacobi exiting on itr=%d of max_itr=%d with error=%f vs threshold=%f\n", itr, JACOBI_MAX_ITR, maxdiff, JACOBI_TOLERANCE);
	error = maxdiff;
	free(M_new);
}
