/* stencil.cc
 */

#include <openacc.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "pch.h"

#define MAX(X,Y) (((X) > (Y)) ? (X) : (Y))

void LaplaceJacobi_naiveACC(float *M, const int b, const int ny, const int nx, const int max_itr, const float threshold){
	/*
	 * Use an iterative Jacobi solver to find the steady-state of
	 * the differential equation of the Laplace equation in 2 dimensions.
	 * M models the initial state of the system and is used to return the
	 * result in-place. M has a border of b entries that aren't updated
	 * by the Jacobi solver. For the iterative Jacobi method, the unknowns
	 * are a flattened version of the interior points. See another source
	 * for more information.
	 */
	int itr = 0;
	float maxdiff = 0.0f;
	float *M_new;

	// Allocate the second version of the M matrix used for the computation
	M_new = (float*)malloc(ny*nx*sizeof(float));

	#pragma acc data copy(M[0:ny*nx]) create(M_new[0:ny*nx])
	{
	do{
		maxdiff = 0.0f;
		itr++;
		#pragma acc parallel copy(maxdiff)
		{
		// Update M_new with M
		#pragma acc loop collapse(2)
		for(int i=b; i<ny-b; i++){
			for(int j=b; j<nx-b; j++){
				M_new[i*nx+j] = 0.25f *(M[(i-1)*nx+j]+M[i*nx+j+1]+ \
							M[(i+1)*nx+j]+M[i*nx+j-1]);
			}
		}

		// Check for convergence while copying values into M
		#pragma acc loop collapse(2) reduction(max:maxdiff)
		for(int i=b; i<ny-b; i++){
			for(int j=b; j<nx-b; j++){
				maxdiff = MAX(fabs(M_new[i*nx+j] - M[i*nx+j]), maxdiff);
				M[i*nx+j] = M_new[i*nx+j];
			}
		}
		} // acc end parallel
	} while(itr < max_itr && maxdiff > threshold);
	} // acc end data
	printf("ACC Jacobi exiting on itr=%d of max_itr=%d with error=%f vs threshold=%f\n", itr, max_itr, maxdiff, threshold);
	free(M_new);
}
