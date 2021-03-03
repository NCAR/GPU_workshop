/* pch.h
 * PreCompiled Header for this lesson
 * Contains the constants, headers, and function prototypes used in other files
 * by: G. Dylan Dickerson (gdicker@ucar.edu)
 */

#ifndef PCH_H_STENCIL
#define PCH_H_STENCIL

// Define the size of the vectors & matrix
#define V_SIZE 1024
#define M_SIZE V_SIZE*V_SIZE

// Convergence tolerance
#define JACOBI_TOLERANCE 1.0E-6F
// Verification tolerance
#define VERIFY_TOL   1.0E-6F

// Convinience struct to return values from LapalceJacobi_x
struct LJ_return
{
    int itr;
    float error;
};

// =======================
// Functions in common.cpp
// =======================
void InitializeMatrixSame(float *array, const int ny, const int nx, const float val, const char *name);
void InitializeMatrixRand(float *array, const int ny, const int nx, const char *name);
void MatrixVerification(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance);
void PrintMatrix(float *matrix, int ny, int nx);
void copyMatrix(float *src, float *dest, const int ny, const int nx);

// =============
// Host routines
// =============
LJ_return LaplaceJacobi_naiveCPU(float *M, const int ny, const int nx);

// ==========================
// Device and OpenACC Routines
// ==========================
LJ_return LaplaceJacobi_naiveACC(float *M, const int ny, const int nx);
#endif // PCH_H_STENCIL
