/* pch.h
 * PreCompiled Header for this lesson
 * Contains the constants, headers, and function prototypes used in other files
 */

#ifndef PCH_H_STENCIL
#define PCH_H_STENCIL

// Define the size of the vectors & matrix
#define V_SIZE 1024
#define M_SIZE V_SIZE*V_SIZE

// InitializeMatrixRand range for the values
#define RANGE_MIN -1
#define RANGE_MAX 1
// InitializeDiagDomMat range for multiplier on diagonal elements
#define DIAG_MIN 5
#define DIAG_MAX 7

// Max number of iterations to run the Jacobi algorithm for
#define JACOBI_MAX_ITR 1000
#define JACOBI_TOLERANCE 1.0E-6F

// =======================
// Functions in common.cpp
// =======================
void InitializeMatrixSame(float *array, const int ny, const int nx, const float val);
void InitializeMatrixRand(float *array, const int ny, const int nx);
void MatrixVerification(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance);
void PrintMatrix(float *matrix, int ny, int nx);
void copyMatrix(float *src, float *dest, const int ny, const int nx);

// =============
// Host routines
// =============
void InitializeDiagDomMat(float *array, const int ny, const int nx);
bool CheckDiagDomMat(float *array, const int ny, const int nx);
float Jacobi_ErrorCalcCPU(const float *A, const float *b, const float *x, const int ny, const int nx);
void Jacobi_naiveCPU(const float *A, const float *b, float *x, const int ny, const int nx, const float threshold);

// ==========================
// Device and Global Routines
// ==========================


#endif // PCH_H_STENCIL
