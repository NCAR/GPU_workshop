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


LJ_return LaplaceJacobi_MPICPU(float *M, const int ny, const int nx,
                               const int rank, const int *neighbors){
/*
 * Performs the same calculations as naiveCPU, but also does a halo exchange
 * at the end of each iteration to update the ghost areas
 */
    // Convenience sizes
    int matsz = ny*nx,
        buffsz_x = nx-2,
        buffsz_y = ny-2;
    int itr = 0;
    float maxdiff = 0.0f, // The error for this process
          g_maxdiff=0.0f; // The max error over all processes
    // Arrays used by this function
    // M_new is the version of M that is updated in the body of the loop before
    // being copied back into M at the end of an iteration
    float *M_new;
    LJ_return ret;

    // MPI Specific Variables
    // Arrays used to send the ghost area values to each neighbor
    float *send_top, *send_right, *send_bot, *send_left;
    // Arrays used to receive the ghost area values from each neighbor
    float *recv_top, *recv_right, *recv_bot, *recv_left;
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
    M_new = (float*)malloc(matsz*sizeof(float));
    send_top = (float*)malloc(buffsz_x*sizeof(float));
    send_right = (float*)malloc(buffsz_y*sizeof(float));
    send_bot = (float*)malloc(buffsz_x*sizeof(float));
    send_left = (float*)malloc(buffsz_y*sizeof(float));
    recv_top = (float*)malloc(buffsz_x*sizeof(float));
    recv_right = (float*)malloc(buffsz_y*sizeof(float));
    recv_bot = (float*)malloc(buffsz_x*sizeof(float));
    recv_left = (float*)malloc(buffsz_y*sizeof(float));

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

        // Perform halo exchange
        if(HasNeighbor(neighbors, DIR_TOP)){
            // Copy the values from the top row of the interior
            for(int j=1; j<nx-1; j++){
                send_top[j-1] = M_new[1*nx+j];
            }

            MPI_Irecv(recv_top, nx-2, MPI_FLOAT, neighbors[DIR_TOP], tag_b, MPI_COMM_WORLD, requestT);
            MPI_Isend(send_top, nx-2, MPI_FLOAT, neighbors[DIR_TOP], tag_t, MPI_COMM_WORLD, requestT+1);
        }
        if(HasNeighbor(neighbors, DIR_BOTTOM)){
            // Copy the values from the bottom row of the interior
            for(int j=1; j<nx-1; j++){
                send_bot[j-1] = M_new[(ny-2)*nx+j];
            }

            MPI_Irecv(recv_bot, nx-2, MPI_FLOAT, neighbors[DIR_BOTTOM], tag_t, MPI_COMM_WORLD, requestB);
            MPI_Isend(send_bot, nx-2, MPI_FLOAT, neighbors[DIR_BOTTOM], tag_b, MPI_COMM_WORLD, requestB+1);
        }
        if(HasNeighbor(neighbors, DIR_RIGHT)){
            for(int i=1; i<ny-1; i++){
                send_right[i-1] = M_new[i*nx+(nx-2)];
            }

            MPI_Irecv(recv_right, nx-2, MPI_FLOAT, neighbors[DIR_RIGHT], tag_l, MPI_COMM_WORLD, requestR);
            MPI_Isend(send_right, nx-2, MPI_FLOAT, neighbors[DIR_RIGHT], tag_r, MPI_COMM_WORLD, requestR+1);
        }
        if(HasNeighbor(neighbors, DIR_LEFT)){
            // Copy the values from the left column of the interior
            for(int i=1; i<ny-1; i++){
                send_left[i-1] = M_new[i*nx+1];
            }

            MPI_Irecv(recv_left, nx-2, MPI_FLOAT, neighbors[DIR_LEFT], tag_r, MPI_COMM_WORLD, requestL);
            MPI_Isend(send_left, nx-2, MPI_FLOAT, neighbors[DIR_LEFT], tag_l, MPI_COMM_WORLD, requestL+1);
        }

        // Wait to receive the values and fill in the correct areas of M_new
        if(HasNeighbor(neighbors, DIR_TOP)){ // Fill the values in the top row
            MPI_Waitall(2, requestT, status);

            for(int j=1; j<nx-1; j++){
                M_new[j] = recv_top[j-1];
            }
        }
        if(HasNeighbor(neighbors, DIR_BOTTOM)){
            MPI_Waitall(2, requestB, status);

            for(int j=1; j<nx-1; j++){ // Fill the values in the bottom row
                M_new[(ny-1)*nx+j] = recv_bot[j-1]; 
            }
        }
        if(HasNeighbor(neighbors, DIR_RIGHT)){
            MPI_Waitall(2, requestR, status);

            for(int i=1; i<ny-1; i++){ // Fill the values in the rightmost column
                M_new[i*nx+(nx-1)] = recv_right[i-1];
            }
        }
        if(HasNeighbor(neighbors, DIR_LEFT)){
            MPI_Waitall(2, requestL, status);

            for(int i=1; i<ny-1; i++){ // Fill the values in the leftmost column
                M_new[i*nx] = recv_left[i-1];
            }
        }
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
    } while(g_maxdiff > JACOBI_TOLERANCE);

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

    // Fill in the return value
    ret.itr = itr;
    ret.error = g_maxdiff;
    return ret;
}


void Verify_MPIvsOneThread(float *global_M, const int g_ny, const int g_nx, float *local_M, const int l_ny, const int l_nx, const int pointsPerDim, const int rank, const int nprocs, int *coords){
    /* Given a version of M that was calculated on 1 thread, coalesce the local versions of M
     * onto thread 0 and verify the results
     */
    if (rank == 0){
        int x_offset, y_offset, startInd;
        MPI_Status recv_stat;
        // Create matrix to collect results into and receive buffer
        float *mpi_M;
        float *recv_buff;
        mpi_M = (float*)malloc(g_ny*g_nx*sizeof(float));
        recv_buff = (float*)malloc(l_ny*l_nx*sizeof(float));

        // Initialize mpi_M and fill with values from rank 0
        printf("Abusing the `InitializeMatrixSame` function from common.cpp\n");
        InitializeMatrixSame(mpi_M, g_ny, g_nx, 0.0f, "mpi_M");
        InitializeMatrixSame(mpi_M, 1, g_nx, 300.0f, "mpi_M");
        for(int i=1; i<l_ny-1; i++){
            for(int j=1; j<l_nx-1; j++){
                mpi_M[i*g_nx+j] = local_M[i*l_nx+j];
            }
        }
        for(int r=1; r<nprocs; r++){
            // Obtain 2D coordinates of the rank
            MPI_Recv(coords, 2, MPI_INT, r, r, MPI_COMM_WORLD, &recv_stat);
            // Find the start index of this block in the global matrix
            x_offset = coords[0]*pointsPerDim;
            y_offset = coords[1]*pointsPerDim+1;
            startInd = y_offset*g_nx+x_offset;

            // Receive the local matrix from this rank
            MPI_Recv(recv_buff, l_ny*l_nx, MPI_FLOAT, r, r, MPI_COMM_WORLD, &recv_stat);
            // Then copy the values over
            for(int i=1; i<l_ny-1; i++){
                for(int j=1; j<l_nx-1; j++){
                    mpi_M[startInd+j] = recv_buff[i*l_nx+j];
                }
                // Advance to the next row
                startInd += g_nx;
            }
        }

        // Check if the matrix passes verification
        MatrixVerification(global_M, mpi_M, g_ny, g_nx, VERIFY_TOL);
        // Cleanup malloc'd memory
        free(mpi_M);
        free(recv_buff);
    }
    else{
        // Using local_M as the send buffer,
        // perform blocking sends tagged with the rank
        int send_stat;
        send_stat = MPI_Send(coords, 2, MPI_INT, 0, rank, MPI_COMM_WORLD);
        send_stat = MPI_Send(local_M, l_ny*l_nx, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);
    }
    // Ensure all ranks leave this routine at the same time
    MPI_Barrier(MPI_COMM_WORLD);
}

