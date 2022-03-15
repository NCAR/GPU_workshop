/* common.cpp
 * To contain functions that are common across all lessons
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define RANGE_MAX 1.0 
#define RANGE_MIN -1.0

/* Sets all values in array equal to val */
void InitializeMatrixSame(float *array, const int ny, const int nx, const float val, const char* name){
    // p serves as another pointer to the start rows within array
    float *p = array;

    for(int i=0; i<ny; i++){
        for(int j=0; j<nx; j++){
            p[j] = val;
        }
        // Advance p to the next row
        p += nx;
    }
    printf("Initialized Matrix %s, %d X %d \n",name, ny, nx);


}

/* Sets all elements of array to a number between [RANGE_MIN,RANGE_MAX] */
void InitializeMatrixRand(float *array, const int ny, const int nx,const char* name){
    // p serves as another pointer to the start rows within array
    float *p = array;

    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            p[j] = ((float)rand() / (RAND_MAX)*(RANGE_MAX - RANGE_MIN) + RANGE_MIN);
        }
        // Advance p to the next row
        p += nx;
    }
    printf("Initialized Random Matrix %s, %d X %d \n",name, ny, nx);
}

/* Compares the matrices element-wise and prints an error message if 
 * the difference between values is above fTolerance
 */
void MatrixVerification(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance){
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
                printf("error: %f > %f", fabs(p[j]-q[j]),fTolerance);
                printf("\t host_C[%d][%d]= %f", i,j, p[j]);
                printf("\t GPU_C[%d][%d]= %f", i,j, q[j]);
                PassFlag=0;
                return;
            }
        }
        p += nx;
        q += nx;
    }
    if(PassFlag)
    {
        printf("Verification passed\n");
    }
}

void PrintMatrix(float *matrix, int ny, int nx){
    if (ny <= 6 && nx <= 6)
    {
        float *p = matrix;

        for (int i = 0; i < ny; i++)
        {
            for (int j = 0; j < nx; j++)
            {
                printf("%f\t",p[j]);
            }
            printf("\n");
            fflush(stdout);
            p += nx;
        }
    }
}

void copyMatrix(float *src, float *dest, const int ny, const int nx){
    float *p = src;
    float *q = dest;
    for(int i=0; i<ny; i++){
        for(int j=0; j<nx; j++){
            q[j] = p[j];
        }
        p += nx;
        q += nx;
    }
}

void InitializeLJMatrix_MPI(float *M, const int ny, const int nx, const int rank, const int *coords){
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
