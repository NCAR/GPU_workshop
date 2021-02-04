//FMA code main 
//D = A*B+C , initially written for matrix mult to be extended to the FMA operation. 

#include "pch.h"

int main()
{
	//input parameters 
	int rows = 1<<10; //1024 elements
	int cols = 1<<10;
	const float A_val = 3.0f; //Arbitrary value to fill the A matrix 
	const float B_val = 2.0f;  
	const float Tol=1.0E-04; //Accuracy tolerance btwn CPU and GPU results

	float *A, *B, *C, *D; 
	float *gpuD;

	//Timing Variables 
	clock_t t0,t1,t2,t3; 
	double t_cpu = 0.0; 
	double t_gpu = 0.0; 

	t0=clock(); //start initialization time

	//Memory Allocation on Host 
	A = new float[rows*cols]; 
	B = new float[rows*cols];
	C = new float[rows*cols];
	D = new float[rows*cols];
	gpuD = new float[rows*cols]; //Space to hold GPU result on CPU while maintaining CPU D for value verification. 	

	//InitializeMatrix
	InitializeMatrixSame(A, rows, cols, A_val); 
	InitializeMatrixSame(B, rows, cols, B_val);
		
	//Multiplication on CPU
	CPU_FMA(A, B,C, D, rows, cols);

	t1 = clock(); 
        t_cpu = ((double)(t1-t0))/CLOCKS_PER_SEC;
        printf("Initialization and execution on CPU executed in %f seconds. \n", t_cpu);	
	t2 = clock(); 
	//Multiplication on GPU 
	gpuFMA(A,B,C,gpuD,rows,cols);	

	t3 = clock();
	t_gpu = ((double)(t3-t2))/CLOCKS_PER_SEC;
        printf("Data Transfer and execution on GPU executed in %f seconds. \n", t_gpu);

	fflush(stdout); 	
	
	//Result Verification
	MatrixVerification(D,gpuD,rows,cols,Tol);	

	//Cleaning up Memory Usage 
	delete[] A; 
	delete[] B; 
	delete[] C; 
	delete[] D; 
	delete[] gpuD; 
	
	return 0; 
}
