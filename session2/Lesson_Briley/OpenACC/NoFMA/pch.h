
#ifndef pch 
#define pch

//Add headers that you want to pre-compile here 
#include <stdio.h>
#include <time.h> 
#include <stdlib.h>
#include <math.h>
#include <openacc.h> 
#include <ctime>
#include <chrono>
#include <ratio> 

//Function Declarations in Common
void InitializeMatrixSame(float *array, const int ny, const int nx, const float val,const char* name);
void InitializeMatrixRand(float *array, const int ny, const int nx,const char* name);
void MatrixVerification(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance);
void PrintMatrix(float *matrix, int ny, int nx);
void copyMatrix(float *src, float *dest, const int ny, const int nx);

//CPU functions
void CPU_FMA(float *A, float *B, float *C, float *D, const int nx, const int ny);
void DisplayElements(float *temp, const int nx);

//Acc Functions
void ACC_FMA(float *A, float *B, float *C, float *accD, const int nx, const int ny);

#endif
