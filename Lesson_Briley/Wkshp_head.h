
#ifndef Wkshp_head 
#define Wkshp_head

//Add headers that you want to pre-compile here 
#include <stdio.h>
#include <time.h> 
//#include <cuda_runtime.h> 

//Function Declarations
void InitializeM(float *array, const int ny, const int nx,const float val);

void cpuMMult(float *A, float *B, float *C, const int nx, const int ny);

//GPU Implementations
extern void gpuMult(float *h_A, float *h_B, float *h_C, const int ny, const int nx);

#endif
