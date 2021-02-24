/* functions.cpp
 * Contains the host-side functions specific to this problem
 * by G. Dylan Dickerson (gdicker@ucar.edu)
 */

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
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
	//printf("CPU Jacobi exiting on itr=%d of max_itr=%d with error=%f vs threshold=%f\n", itr, max_itr, maxdiff, threshold);
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

void MatrixVerification_MPI(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance, int rank){
	// Pointers for rows in each matrix
	float *p = hostC;
	float *q = gpuC;
        bool PassFlag = 1;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			if (fabs(p[j] - q[j]) > fTolerance)
			{
				printf("Rank:%d error: %f > %f", rank, fabs(p[j]-q[j]),fTolerance);
				printf("\tRank:%d host_M[%d][%d]= %f", rank, i,j, p[j]);
				printf("\tRank:%d GPU_M[%d][%d]= %f", rank, i,j, q[j]);
                                PassFlag=0;
				return;
			}
		}
		p += nx;
		q += nx;
	}
        if(PassFlag)
	{
		printf("Rank:%d Verification passed\n", rank);
        }
}

void LaplaceJacobi_MPICPU(float *M, const int ny, const int nx, const int max_itr, const float threshold, const int rank, const int *coord, const int *neighbors){
/*
 * Performs the same calculations as naiveCPU, but also does a halo exchange
 * at the end of each iteration to update the ghost areas
 */

	int itr = 0;
	float maxdiff = 0.0f, // The error for this process
	      g_maxdiff=0.0f; // The max error over all processes
	// Arrays used by this function
	// M_new is the version of M that is updated in the body of the loop before
	// being copied back into M at the end of an iteration
	float *M_new;
	// Change to a single send buffer?
	// Arrays used to send the ghost area values to each neighbor
	float *send_top, *send_right, *send_bot, *send_left;
	// Change to a single receive buffer?
	// Arrays used to receive the ghost area values from each neighbor
	float *recv_top, *recv_right, *recv_bot, *recv_left;

	// MPI Specific Variables
	// Holds the statuses returned by MPI_Waitall related to a Irecv/Isend pair
	MPI_Status status[2];
	// Groups Irecv/Isend calls together from the sender's perspective and are
	// used by MPI_Waitall before putting received values into M_new
	// (e.g. requestR are the requests for receiving and sending to its right neighbor)
	MPI_Request requestT[2], requestR[2], requestB[2], requestL[2];
	// The (optional) tags for the MPI Isend/Irecv.
	// Tags are relative to the sender (e.g. a process sending data to its 
	// left neighbor uses tag_l in the Isend and the neighbor will use tag_l in its Irecv)
	int tag_t = DIR_TOP, tag_b=DIR_BOTTOM, tag_r=DIR_RIGHT, tag_l=DIR_LEFT;

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

	// Make M_new a copy of M, this helps for the last loop inside the do-while
	std::copy(M, M+(ny*nx), M_new);

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
		//printf("Rank:%d finished jacobi update in M_new, starting halo exchange\n",rank); fflush(stdout);
		
		// Perform halo exchange
		if(HasNeighbor(neighbors, DIR_TOP)){
			//printf("Rank:%d Start top exchange\n", rank); fflush(stdout);
			
			// Copy the values from the top row of the interior
			for(int j=1; j<nx-1; j++){
				send_top[j-1] = M_new[1*nx+j];
			}
			
			//printf("Rank:%d filled top send buffer\n", rank); fflush(stdout);
			//printf("\tRank:%d send_top\n",rank);fflush(stdout);
			//PrintMatrix(send_top, 1, nx-2); fflush(stdout);
			
			MPI_Irecv(recv_top, nx-2, MPI_FLOAT, neighbors[DIR_TOP], tag_b, MPI_COMM_WORLD, requestT);
			MPI_Isend(send_top, nx-2, MPI_FLOAT, neighbors[DIR_TOP], tag_t, MPI_COMM_WORLD, requestT+1);
			
			//printf("Rank:%d End top exchange\n", rank);fflush(stdout);
		}
		if(HasNeighbor(neighbors, DIR_BOTTOM)){
			//printf("Rank:%d Start bottom exchange\n", rank);fflush(stdout);

			// Copy the values from the bottom row of the interior
			for(int j=1; j<nx-1; j++){
				send_bot[j-1] = M_new[(ny-2)*nx+j];
			}

			//printf("Rank:%d filled bottom send buffer\n", rank); fflush(stdout);
			//printf("\tRank:%d send_bot\n",rank);fflush(stdout);
			//PrintMatrix(send_bot, 1, nx-2); fflush(stdout);

			MPI_Irecv(recv_bot, nx-2, MPI_FLOAT, neighbors[DIR_BOTTOM], tag_t, MPI_COMM_WORLD, requestB);
			MPI_Isend(send_bot, nx-2, MPI_FLOAT, neighbors[DIR_BOTTOM], tag_b, MPI_COMM_WORLD, requestB+1);
			
			//printf("Rank:%d End bottom exchange\n", rank);fflush(stdout);
		}
		if(HasNeighbor(neighbors, DIR_RIGHT)){
			//printf("Rank:%d Start right exchange\n", rank);fflush(stdout);

			// Copy the values from the right column of the interior
			for(int i=1; i<ny-1; i++){
				send_right[i-1] = M_new[i*nx+(nx-2)];
			}

			//printf("Rank:%d filled right send buffer\n", rank); fflush(stdout);
			//printf("\tRank:%d send_right\n",rank);fflush(stdout);
			//PrintMatrix(send_right, 1, nx-2); fflush(stdout);

			MPI_Irecv(recv_right, nx-2, MPI_FLOAT, neighbors[DIR_RIGHT], tag_l, MPI_COMM_WORLD, requestR);
			MPI_Isend(send_right, nx-2, MPI_FLOAT, neighbors[DIR_RIGHT], tag_r, MPI_COMM_WORLD, requestR+1);

			//printf("Rank:%d End right exchange\n", rank);fflush(stdout);
		}
		if(HasNeighbor(neighbors, DIR_LEFT)){
			//printf("Rank:%d Start left exchange\n", rank);fflush(stdout);

			// Copy the values from the left column of the interior
			for(int i=1; i<ny-1; i++){
				send_left[i-1] = M_new[i*nx+1];
			}

			//printf("Rank:%d filled left send buffer\n", rank); fflush(stdout);
			//printf("\tRank:%d send_left\n",rank);fflush(stdout);
			//PrintMatrix(send_left, 1, nx-2); fflush(stdout);

			MPI_Irecv(recv_left, nx-2, MPI_FLOAT, neighbors[DIR_LEFT], tag_r, MPI_COMM_WORLD, requestL);
			MPI_Isend(send_left, nx-2, MPI_FLOAT, neighbors[DIR_LEFT], tag_l, MPI_COMM_WORLD, requestL+1);

			//printf("Rank:%d End left exchange\n", rank);fflush(stdout);
		}

		// Wait for the values and fill in the correct areas of M_new
		if(HasNeighbor(neighbors, DIR_TOP)){ // Fill the values in the top row
			MPI_Waitall(2, requestT, status);

			//printf("Rank:%d using recv_top to fill M_new\n", rank); fflush(stdout);
			//printf("Rank:%d recv_top\n", rank);fflush(stdout);
			//PrintMatrix(recv_top, 1, nx-2); fflush(stdout);

			for(int j=1; j<nx-1; j++){
				M_new[j] = recv_top[j-1];
			}

			//printf("Rank:%d filled M_new with recv_top\n", rank); fflush(stdout);
		}
		if(HasNeighbor(neighbors, DIR_BOTTOM)){
			MPI_Waitall(2, requestB, status);

			//printf("Rank:%d using recv_bot to fill M_new\n", rank); fflush(stdout);
			//printf("Rank:%d recv_bot\n", rank);fflush(stdout);
			//PrintMatrix(recv_bot, 1, nx-2); fflush(stdout);

			for(int j=1; j<nx-1; j++){ // Fill the values in the bottom row
				M_new[(ny-1)*nx+j] = recv_bot[j-1];	
			}

			//printf("Rank:%d filled M_new with recv_bot\n", rank); fflush(stdout);
		}
		if(HasNeighbor(neighbors, DIR_RIGHT)){
			MPI_Waitall(2, requestR, status);

			//printf("Rank:%d using recv_right to fill M_new\n", rank); fflush(stdout);
			//printf("Rank:%d recv_right\n", rank);fflush(stdout);
			//PrintMatrix(recv_right, 1, nx-2); fflush(stdout);

			for(int i=1; i<ny-1; i++){ // Fill the values in the rightmost column
				M_new[i*nx+(nx-1)] = recv_right[i-1];
			}

			//printf("Rank:%d filled M_new with recv_right\n", rank); fflush(stdout);
		}
		if(HasNeighbor(neighbors, DIR_LEFT)){
			MPI_Waitall(2, requestL, status);

			//printf("Rank:%d using recv_left to fill M_new\n", rank); fflush(stdout);
			//printf("Rank:%d recv_left\n", rank);fflush(stdout);
			//PrintMatrix(recv_left, 1, nx-2); fflush(stdout);

			for(int i=1; i<ny-1; i++){ // Fill the values in the leftmost column
				M_new[i*nx] = recv_left[i-1];
			}

			//printf("Rank:%d filled M_new with recv_left\n", rank); fflush(stdout);
		}

		//printf("Rank:%d End halo exchange\n", rank); fflush(stdout);
		// End the halo exchange section

		// Check for convergence while copying values into M
		for(int i=0; i<ny; i++){
			for(int j=0; j<nx; j++){
				maxdiff = MAX(fabs(M_new[i*nx+j] - M[i*nx+j]), maxdiff);
				M[i*nx+j] = M_new[i*nx+j];
			}
		}
		// Find the global max difference. Have each process exit when the global error is low enough
		MPI_Allreduce(&maxdiff, &g_maxdiff, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
		
		//printf("Rank:%d Completed transfer and iteration %d\n",rank, itr); fflush(stdout);
	} while(itr < max_itr && g_maxdiff > threshold);

	//printf("Rank:%d MPI-CPU Jacobi exiting on itr=%d of max_itr=%d with error=%f vs threshold=%f\n", rank, itr, max_itr, maxdiff, threshold);

	// Free malloc'ed memory
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
