/* stencil.cc
 * by: G. Dylan Dickerson (gdicker@ucar.edu)
 */

#include <openacc.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <mpi.h>
#include <openacc.h>
#include "pch.h"

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


#pragma acc enter data copyin(M[0:ny*nx], M_new[0:ny*nx])
#pragma acc enter data create(send_top[0:(nx-2)], recv_top[0:nx-2])
#pragma acc enter data create(send_right[0:(ny-2)], recv_right[0:ny-2])
#pragma acc enter data create(send_bot[0:(nx-2)], recv_bot[0:nx-2])
#pragma acc enter data create(send_left[0:(ny-2)], recv_left[0:ny-2])

    // Make M_new a copy of M, this helps for the last loop inside the do-while
#pragma acc parallel loop collapse(2) present(M[0:matsz], M_new[0:matsz])
    for(int i=0; i<ny; i++){
        for(int j=0; j<nx; j++){
            M_new[i*nx+j] = M[i*nx+j];
        }
    }

    do {
        maxdiff = 0.0f;
        itr++;

        // Update M_new with M
#pragma acc parallel loop collapse(2) present(M[0:matsz], M_new[0:matsz])
        for(int i=1; i<ny-1; i++){
            for(int j=1; j<nx-1; j++){
                M_new[i*nx+j] = 0.25f *(M[(i-1)*nx+j]+M[i*nx+j+1]+ \
                                        M[(i+1)*nx+j]+M[i*nx+j-1]);
            }
        }

        // Perform halo exchange
        if(HasNeighbor(neighbors, DIR_TOP)){
            // Copy the values from the top row of the interior
#pragma acc parallel loop present(send_top[0:buffsz_x], M_new[0:matsz])
            for(int j=1; j<nx-1; j++){
                send_top[j-1] = M_new[1*nx+j];
            }
#pragma acc host_data use_device(recv_top, send_top)
{
            MPI_Irecv(recv_top, nx-2, MPI_FLOAT, neighbors[DIR_TOP], tag_b, MPI_COMM_WORLD, requestT);
            MPI_Isend(send_top, nx-2, MPI_FLOAT, neighbors[DIR_TOP], tag_t, MPI_COMM_WORLD, requestT+1);
}
        }
        if(HasNeighbor(neighbors, DIR_BOTTOM)){
            // Copy the values from the bottom row of the interior
#pragma acc parallel loop  present(send_bot[0:buffsz_x], M_new[0:matsz])
            for(int j=1; j<nx-1; j++){
                send_bot[j-1] = M_new[(ny-2)*nx+j];
            }
#pragma acc host_data use_device(recv_bot, send_bot)
{
            MPI_Irecv(recv_bot, nx-2, MPI_FLOAT, neighbors[DIR_BOTTOM], tag_t, MPI_COMM_WORLD, requestB);
            MPI_Isend(send_bot, nx-2, MPI_FLOAT, neighbors[DIR_BOTTOM], tag_b, MPI_COMM_WORLD, requestB+1);
}
        }
        if(HasNeighbor(neighbors, DIR_RIGHT)){
            // Copy the values from the right column of the interior
#pragma acc parallel loop  present(send_right[0:buffsz_y], M_new[0:matsz])
            for(int i=1; i<ny-1; i++){
                send_right[i-1] = M_new[i*nx+(nx-2)];
            }
#pragma acc host_data use_device(recv_right, send_right)
{
            MPI_Irecv(recv_right, nx-2, MPI_FLOAT, neighbors[DIR_RIGHT], tag_l, MPI_COMM_WORLD, requestR);
            MPI_Isend(send_right, nx-2, MPI_FLOAT, neighbors[DIR_RIGHT], tag_r, MPI_COMM_WORLD, requestR+1);
}
        }
        if(HasNeighbor(neighbors, DIR_LEFT)){
            // Copy the values from the left column of the interior
#pragma acc parallel loop present(send_left[0:buffsz_y], M_new[0:matsz])  
            for(int i=1; i<ny-1; i++){
                send_left[i-1] = M_new[i*nx+1];
            }
#pragma acc host_data use_device(recv_left, send_left)
{
            MPI_Irecv(recv_left, nx-2, MPI_FLOAT, neighbors[DIR_LEFT], tag_r, MPI_COMM_WORLD, requestL);
            MPI_Isend(send_left, nx-2, MPI_FLOAT, neighbors[DIR_LEFT], tag_l, MPI_COMM_WORLD, requestL+1);
}
        }

        // Wait to receive the values and fill in the correct areas of M_new
        if(HasNeighbor(neighbors, DIR_TOP)){ // Fill the values in the top row
            MPI_Waitall(2, requestT, status);
#pragma acc parallel loop present(recv_top[0:buffsz_x], M_new[0:matsz])  
            for(int j=1; j<nx-1; j++){
                M_new[j] = recv_top[j-1];
            }
        }
        if(HasNeighbor(neighbors, DIR_BOTTOM)){
            MPI_Waitall(2, requestB, status);
#pragma acc parallel loop present(recv_bot[0:buffsz_x], M_new[0:matsz]) 
            for(int j=1; j<nx-1; j++){ // Fill the values in the bottom row
                M_new[(ny-1)*nx+j] = recv_bot[j-1]; 
            }
        }
        if(HasNeighbor(neighbors, DIR_RIGHT)){
            MPI_Waitall(2, requestR, status);
#pragma acc parallel loop present(recv_right[0:buffsz_y], M_new[0:matsz])  
            for(int i=1; i<ny-1; i++){ // Fill the values in the rightmost column
                M_new[i*nx+(nx-1)] = recv_right[i-1];
            }
        }
        if(HasNeighbor(neighbors, DIR_LEFT)){
            MPI_Waitall(2, requestL, status);
#pragma acc parallel loop present(recv_left[0:buffsz_y], M_new[0:matsz])  
            for(int i=1; i<ny-1; i++){ // Fill the values in the leftmost column
                M_new[i*nx] = recv_left[i-1];
            }
        }
        // End the halo exchange section

        // Check for convergence while copying values into M
#pragma acc parallel loop collapse(2) reduction(max:maxdiff) \
 present(M[0:matsz],M_new[0:matsz]) copy(maxdiff)
        for(int i=0; i<ny; i++){
            for(int j=0; j<nx; j++){
                maxdiff = MAX(fabs(M_new[i*nx+j] - M[i*nx+j]), maxdiff);
                M[i*nx+j] = M_new[i*nx+j];
            }
        }
        // Find the global max difference. Have each process exit when the global error is low enough
        MPI_Allreduce(&maxdiff, &g_maxdiff, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    } while(g_maxdiff > JACOBI_TOLERANCE);

#pragma acc exit data copyout(M[0:matsz]) delete(M_new[0:matsz])
#pragma acc exit data delete(send_top[0:buffsz_x], send_right[0:buffsz_y])
#pragma acc exit data delete(send_bot[0:buffsz_x], send_left[0:buffsz_y])
#pragma acc exit data delete(recv_top[0:buffsz_x], recv_right[0:buffsz_y])
#pragma acc exit data delete(recv_bot[0:buffsz_x], recv_left[0:buffsz_y])

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
