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

LJ_return LaplaceJacobi_naiveCPU(float *M, const int ny, const int nx){
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
    LJ_return ret;

    // Allocate the second version of the M matrix used for the computation
    M_new = (float*)malloc(ny*nx*sizeof(float));

    do {
        maxdiff = 0.0f;
        itr++;
        // Update M_new with M
        for(int i=1; i<ny-1; i++){
            for(int j=1; j<nx-1; j++){
                M_new[i*nx+j] = 0.25f *(M[(i-1)*nx+j]+M[i*nx+j+1]+ \
                            M[(i+1)*nx+j]+M[i*nx+j-1]);
            }
        }

        // Check for convergence while copying values into M
        for(int i=1; i<ny-1; i++){
            for(int j=1; j<nx-1; j++){
                maxdiff = MAX(fabs(M_new[i*nx+j] - M[i*nx+j]), maxdiff);
                M[i*nx+j] = M_new[i*nx+j];
            }
        }
    } while(maxdiff > JACOBI_TOLERANCE);
    
    // Free malloc'd memory
    free(M_new);
    
    // Fill in the return value
    ret.itr = itr;
    ret.error = maxdiff;
    return ret;
}
