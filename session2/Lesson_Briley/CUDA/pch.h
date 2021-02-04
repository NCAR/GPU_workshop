
#ifndef pch 
#define pch

//Add headers that you want to pre-compile here 
#include <stdio.h>
#include <time.h> 
#include <stdlib.h>
#include <math.h>

//Function Declarations in Common
void InitializeMatrixSame(float *array, const int ny, const int nx, const float val);
void InitializeMatrixRand(float *array, const int ny, const int nx);
void MatrixVerification(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance);

//CPU functions
void CPU_FMA(float *A, float *B, float *C, float *D, const int nx, const int ny);

//GPU functions
extern void gpuFMA(float *h_A, float *h_B, float *h_C, float *h_D, const int ny, const int nx);

#endif
