/* stencil.cc
 * by: G. Dylan Dickerson (gdicker@ucar.edu)
 */

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <mpi.h>
#include "pch.h"

// TODO: Include openacc.h for the acc_* functions

#define MAX(X,Y) (((X) > (Y)) ? (X) : (Y))

void mapGPUToMPIRanks(int rank){
    // Get device count
    const int num_dev = acc_get_num_devices(acc_device_nvidia);
    // Pin this rank to a GPU
    const int dev_id = rank % num_dev;
    acc_set_device_num(dev_id, acc_device_nvidia);
}


LJ_return LaplaceJacobi_MPIACC(float *M, const int ny, const int nx,
                               const int rank, const int *neighbors){
/*
 * Performs the same calculations as naiveCPU, but also does a halo exchange
 * at the end of each iteration to update the ghost areas and uses OpenACC pragmas
 */
    // Convenience sizes
    int matsz = ny*nx,
        buffsz_x = nx;
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
    float *send_top, *send_bot;
    // Arrays used to receive the ghost area values from each neighbor
    float *recv_top, *recv_bot;
    // Holds the statuses returned by MPI_Waitall related to a Irecv/Isend pair
    MPI_Status status[2];
    // Groups Irecv/Isend calls together from the sender's perspective and are
    // used by MPI_Waitall before putting received values into M_new
    // (e.g. requestR are the requests for receiving and sending to its right neighbor)
    MPI_Request requestT[2], requestB[2];
    // The (optional) tags for the MPI Isend/Irecv.
    // Tags are relative to the sender (e.g. a process sending data to its 
    // left neighbor uses tag_l in the Isend and the neighbor will use tag_l in its Irecv)
    int tag_t = DIR_TOP, tag_b=DIR_BOTTOM;

    // Allocate local arrays
    M_new = (float*)malloc(matsz*sizeof(float));
    send_top = (float*)malloc(buffsz_x*sizeof(float));
    send_bot = (float*)malloc(buffsz_x*sizeof(float));
    recv_top = (float*)malloc(buffsz_x*sizeof(float));
    recv_bot = (float*)malloc(buffsz_x*sizeof(float));


// TODO: Create an unstructured data region copyin M and M_new
// as well as create the send_* and recv_* variables



// TODO: Parallelize this copy loop
    // Make M_new a copy of M, this helps for the last loop inside the do-while
    for(int i=0; i<ny; i++){
        for(int j=0; j<nx; j++){
            M_new[i*nx+j] = M[i*nx+j];
        }
    }

    do {
        maxdiff = 0.0f;
        itr++;

        // Update M_new with M
// TODO: Parallelize the update loop
        for(int i=1; i<ny-1; i++){
            for(int j=1; j<nx-1; j++){
                M_new[i*nx+j] = 0.25f *(M[(i-1)*nx+j]+M[i*nx+j+1]+ \
                                        M[(i+1)*nx+j]+M[i*nx+j-1]);
            }
        }

        // Perform halo exchange
        if(HasNeighbor(neighbors, DIR_TOP)){
            // Copy the values from the top row of the interior
// TODO: Parallelize this loop copying into the send buffer
            for(int j=0; j<nx; j++){
                send_top[j] = M_new[1*nx+j];
            }
// TODO: Add host_data construct
            MPI_Irecv(recv_top, buffsz_x, MPI_FLOAT, neighbors[DIR_TOP], tag_b, MPI_COMM_WORLD, requestT);
            MPI_Isend(send_top, buffsz_x, MPI_FLOAT, neighbors[DIR_TOP], tag_t, MPI_COMM_WORLD, requestT+1);
        }
        if(HasNeighbor(neighbors, DIR_BOTTOM)){
            // Copy the values from the bottom row of the interior
// TODO: Parallelize this loop copying into the send buffer
            for(int j=0; j<nx; j++){
                send_bot[j] = M_new[(ny-2)*nx+j];
            }
// TODO: Add host_data construct
            MPI_Irecv(recv_bot, buffsz_x, MPI_FLOAT, neighbors[DIR_BOTTOM], tag_t, MPI_COMM_WORLD, requestB);
            MPI_Isend(send_bot, buffsz_x, MPI_FLOAT, neighbors[DIR_BOTTOM], tag_b, MPI_COMM_WORLD, requestB+1);
        }

        // Wait to receive the values and fill in the correct areas of M_new
        if(HasNeighbor(neighbors, DIR_TOP)){ // Fill the values in the top row
            MPI_Waitall(2, requestT, status);
// TODO: Parallelize this loop copying into the border area
            for(int j=1; j<nx-1; j++){
                M_new[j] = recv_top[j-1];
            }
        }
        if(HasNeighbor(neighbors, DIR_BOTTOM)){
            MPI_Waitall(2, requestB, status);
// TODO: Parallelize this loop copying into the border area
            for(int j=1; j<nx-1; j++){ // Fill the values in the bottom row
                M_new[(ny-1)*nx+j] = recv_bot[j-1]; 
            }
        }
        // End the halo exchange section

        // Check for convergence while copying values into M
// TODO: Parallelize the convergence loop, apply a max reduction
        for(int i=0; i<ny; i++){
            for(int j=0; j<nx; j++){
                maxdiff = MAX(fabs(M_new[i*nx+j] - M[i*nx+j]), maxdiff);
                M[i*nx+j] = M_new[i*nx+j];
            }
        }
        // Find the global max difference. Have each process exit when the global error is low enough
        MPI_Allreduce(&maxdiff, &g_maxdiff, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    } while(g_maxdiff > JACOBI_TOLERANCE && itr < JACOBI_MAX_ITR);

// TODO: Add pragmas to copyout data and delete other arrays for unstructured data region

    // Free malloc'ed memory
    free(M_new);
    free(send_top);
    free(send_bot);
    free(recv_top);
    free(recv_bot);

    // Fill in the return value
    ret.itr = itr;
    ret.error = g_maxdiff;
    return ret;
}
