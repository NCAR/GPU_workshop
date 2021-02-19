/* functions.cpp
 * Contains the host-side functions specific to this problem
 * by G. Dylan Dickerson (gdicker@ucar.edu)
 */

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "pch.h"

#define MAX(X,Y) (((X) > (Y)) ? (X) : (Y))

void LaplaceJacobi_naiveCPU(float *M, const int b, const int ny, const int nx, const int
max_itr, const float threshold){
	// Use an iterative Jacobi solver to find the steady-state of the differential equation
	// of the Laplace eqution in 2 dimensions. M models the initial state of the system and
	// is used to return the result in-place. M has a border of b entries that aren't updated
	// by the Jacobi solver. For the iterative Jacobi method, the unknowns are a flattened
	// version of the interior points. See another source for more information
	//
	// The result is solving a system of the form 
	// 	M[i][j] = 1/4(M[i-1][j] + M[i][j+1] + M[i+1][j] + M[i][j-1])
	int itr = 0;
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
	} while(itr < max_itr && maxdiff > threshold);
	printf("CPU Jacobi exiting on itr=%d of max_itr=%d with error=%f vs threshold=%f\n", itr, max_itr, maxdiff, threshold);
	free(M_new);
}

void InitializeMatrix_MPI(float *M, const int ny, const int nx, const int rank, const int *coords){
/*
 * Use the coordinates of the process to determine if it is in the top row and
 * set the top row of the local matrix to 300.0f if so. Otherwise all values are
 * filled with 0.0f 
 */
	int startRow = 0;
	if(coords[1] == 0){ // if in the first row
		for(int j=0; j<nx; j++){
			M[j] = 300.0f;
		}
		startRow = 1;
	}
	for(int i=startRow; i<ny; i++){
		for(int j=0; j<nx; j++){
			M[i*nx+j] = 0.0f;
		}
	}
}

void LaplaceJacobi_MPICPU(float *M, const int ny, const int nx, const int max_itr, const float threshold, const int rank, const int *coord, const int *neighbors){
/*
 * Performs the same calculations as naiveCPU, but also does a halo exchange
 * at the end of each iteration to update the ghost areas
 */

	int itr = 0;
	float maxdiff = 0.0f;
	float *M_new;
	float *send_top, *send_right, *send_bot, *send_left;
	float *recv_top, *recv_right, *recv_bot, *recv_left;
	MPI_Status status;
	MPI_Request requestST, requestSR, requestSB, requestSL,
		    requestRT, requestRR, requestRB, requestRL;

	// Allocate local arrays
	M_new = (float*)malloc(ny*nx*sizeof(float));
	send_top = (float*)malloc((nx-2)*sizeof(float));
	send_right = (float*)malloc((ny-2)*sizeof(float));
	send_bot = (float*)malloc((nx-2)*sizeof(float));
	send_left = (float*)malloc((ny-2)*sizeof(float));
	recv_top = (float*)malloc((nx-2)*sizeof(float));
	recv_right = (float*)malloc((ny-2)*sizeof(float));
	recv_bot = (float*)malloc((nx-2)*sizeof(float));
	recv_left = (float*)malloc((ny-2)*sizeof(float));

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
		
		// Perform halo exchange
		if(HasNeighbor(neighbors, DIR_TOP)){
			// Copy the values from the top row of the interior
			for(int j=1; j<nx-1; j++){
				send_top[j-1] = M_new[1*nx+j];
			}
			MPI_Isend(&send_top, nx-2, MPI_FLOAT, neighbors[DIR_TOP], 0, MPI_COMM_WORLD, &requestST);
			MPI_Irecv(&recv_top, nx-2, MPI_FLOAT, neighbors[DIR_TOP], 0, MPI_COMM_WORLD, &requestRT);
		}
		if(HasNeighbor(neighbors, DIR_RIGHT)){
			// Copy the values from the right column of the interior
			for(int i=1; i<ny-1; i++){
				send_right[i-1] = M_new[i*nx+(nx-2)];
			}
			MPI_Isend(&send_right, ny-2, MPI_FLOAT, neighbors[DIR_RIGHT], 0, MPI_COMM_WORLD, &requestSR);
			MPI_Irecv(&recv_right, ny-2, MPI_FLOAT, neighbors[DIR_RIGHT], 0, MPI_COMM_WORLD, &requestRR);
		}
		if(HasNeighbor(neighbors, DIR_BOTTOM)){
			// Copy the values from the bottom row of the interior
			for(int j=1; j<nx-1; j++){
				send_bot[j-1] = M_new[(ny-2)*nx+j];
			}
			MPI_Isend(&send_bot, nx-2, MPI_FLOAT, neighbors[DIR_BOTTOM], 0, MPI_COMM_WORLD, &requestSB);
			MPI_Irecv(&recv_bot, nx-2, MPI_FLOAT, neighbors[DIR_BOTTOM], 0, MPI_COMM_WORLD, &requestRB);
		}
		if(HasNeighbor(neighbors, DIR_LEFT)){
			// Copy the values from the left column of the interior
			for(int i=1; i<ny-1; i++){
				send_left[i-1] = M_new[i*nx+1];
			}
			MPI_Isend(&send_left, ny-2, MPI_FLOAT, neighbors[DIR_LEFT], 0, MPI_COMM_WORLD, &requestSL);
			MPI_Irecv(&recv_left, ny-2, MPI_FLOAT, neighbors[DIR_LEFT], 0, MPI_COMM_WORLD, &requestRL);
		}

		// Wait for the values and fill the correct areas of M_new
		if(HasNeighbor(neighbors, DIR_TOP)){ // Fill the values in the top row
			MPI_Wait(&requestRT, &status);
			for(int j=1; j<nx-1; j++){
				M_new[j] = recv_top[j-1];
			}
		}
		if(HasNeighbor(neighbors, DIR_RIGHT)){
			MPI_Wait(&requestRR, &status);
			for(int i=1; i<ny-1; i++){ // Fill the values in the rightmost column
				M_new[i*nx+(nx-1)] = recv_right[i-1];
			}
		}
		if(HasNeighbor(neighbors, DIR_BOTTOM)){
			MPI_Wait(&requestRB, &status);
			for(int j=1; j<nx-1; j++){ // Fill the values in the bottom row
				M_new[(ny-1)*nx+j] = recv_bot[j-1];	
			}
		}
		if(HasNeighbor(neighbors, DIR_LEFT)){
			MPI_Wait(&requestRL, &status);
			for(int i=1; i<ny-1; i++){ // Fill the values in the leftmost column
				M_new[i*nx] = recv_left[i-1];
			}
		}

		// End the halo exchange section

		// Check for convergence while copying values into M
		for(int i=1; i<ny-1; i++){
			for(int j=1; j<nx-1; j++){
				maxdiff = MAX(fabs(M_new[i*nx+j] - M[i*nx+j]), maxdiff);
				M[i*nx+j] = M_new[i*nx+j];
			}
		}
	} while(itr < max_itr && maxdiff > threshold);
	printf("Rank:%d CPU Jacobi exiting on itr=%d of max_itr=%d with error=%f vs threshold=%f\n", rank, itr, max_itr, maxdiff, threshold);
	free(M_new);
	free(send_top);
	free(send_right);
	free(send_bot);
	free(send_left);
	free(recv_top);
	free(recv_right);
	free(recv_bot);
	free(recv_left);	
}
