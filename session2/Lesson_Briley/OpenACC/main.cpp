//FMA code main 
//D = A*B+C 

#include "pch.h"

int main()
{
	//input parameters 
	int rows = 1<<10; //1024 elements
	int cols = 1<<10;
	const float A_val = 3.0f; //Arbitrary value to fill the A matrix 
	const float B_val = 2.0f;  
	const float C_val = 1.0f; 
	const float Tol=1.0E-04; //Accuracy tolerance btwn CPU and GPU results

	float *A, *B, *C, *D; 
	float *accD;

	//Timing Variables 
	clock_t t0,t1,t2,t3; 
	double t_cpu = 0.0; 
	double t_acc = 0.0f; 

	t0=clock(); //start initialization time

	//Memory Allocation on Host 
	A = new float[rows*cols]; 
	B = new float[rows*cols];
	C = new float[rows*cols];
	D = new float[rows*cols];
	accD = new float[rows*cols]; //Space to hold Open Acc GPU result on CPU while maintaining CPU D for value verification.	

	//InitializeMatrix
	InitializeMatrixSame(A, rows, cols, A_val); 
	InitializeMatrixSame(B, rows, cols, B_val); 
	InitializeMatrixSame(C, rows, cols, C_val);
	
	//FMA execution on CPU
	CPU_FMA(A, B,C, D, rows, cols);

	t1 = clock(); 
        t_cpu = ((double)(t1-t0))/CLOCKS_PER_SEC;
        printf("Initialization and execution on CPU executed in %f seconds. \n", t_cpu);	
	t2 = clock(); 
	
	//Directive to transfer data to GPU
	#pragma acc enter data copyin(A[:rows*cols],B[:rows*cols],C[:rows*cols])
	//OpenACC FMA execution
	ACC_FMA(A,B,C,accD,rows,cols); 

	t3 = clock(); 
	t_acc = ((double)(t3-t2))/CLOCKS_PER_SEC;
        printf("Data Transfer and execution on GPU with OpenAcc executed in %f seconds. \n", t_acc);

	//Result Verification
        printf("CPU to OpenACC GPU result confirmation: \t");
        MatrixVerification(D,accD,rows,cols,Tol);

	//Cleaning up Memory Usage 
	delete[] A; 
	delete[] B; 
	delete[] C; 
	delete[] D; 
	delete[] accD; 
	
	return 0; 
}
