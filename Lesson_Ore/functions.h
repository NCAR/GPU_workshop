//This file is used for functions

//Add headers that you want to pre-compile here 
#include <stdio.h>
#include <time.h> 
#define MATMUL_TOL 1.0E-6F
#define MATMUL_A_VAL 3.0F
#define MATMUL_B_VAL 2.0F

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
        msg, cudaGetErrorString(__err), \
        __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n"); \
      exit(1); \
    } \
  } while (0)

void cpuMatmul(const float *A, const float *B, float *C, const int m, const int p, const int q);

extern void gpuMult(float *h_A, float *h_B, float *h_check, const int m, const int n, const int p, const int q, const int block_size);

void InitializeMatrixSame(float *array, const int ny, const int nx, const float val);

void InitializeMatrixRand(float *array, const int ny, const int nx);

void MatrixVerification(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance);

void PrintMatrix(float *matrix, int ny, int nx);

void copyMatrix(float *src, float *dest, const int ny, const int nx);

void OpenAccMult(const float *A, const float *B, float *C, const int m, const int p, const int q); 
