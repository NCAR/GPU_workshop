//FMA code main 
//D = A*B+C 

#include "pch.h"
//#include <chrono> 
//#include <ctime> 
//#include <ratio> 

int main()
{
	using namespace std::chrono; 

	//input parameters 
	int rows = 1<<10; //1024 elements
	int cols = 1<<10;
	const float A_val = 3.3f; //Arbitrary value to fill the A matrix 
	const float B_val = 2.2f;  
	const float Tol=1.0E-04; //Accuracy tolerance btwn CPU and GPU results

	float *A, *B, *C, *D, *accD; 

	//Timing Variables 
	high_resolution_clock::time_point t0,t1;
	duration<double> t_sum;

	t0=high_resolution_clock::now(); //start initialization time

	//Memory Allocation on Host 
	A = new float[rows*cols]; 
	B = new float[rows*cols];
	C = new float[rows*cols];
	D = new float[rows*cols];
	accD = new float[rows*cols]; 	

	//InitializeMatrix
	InitializeMatrixSame(A, rows, cols, A_val,"h_A"); 
	InitializeMatrixSame(B, rows, cols, B_val,"h_B"); 
	InitializeMatrixRand(C, rows, cols,"h_C");
	
	t1= high_resolution_clock::now();
	t_sum = duration_cast<duration<double>>(t1-t0);
	printf("Initialization took %f seconds. Begin CPU compute. \n", t_sum.count()); 

	//FMA execution on host(CPU)
	t0 = high_resolution_clock::now(); 
	CPU_FMA(A, B,C, D, rows, cols);
	t1 = high_resolution_clock::now(); 
	
	t_sum = duration_cast<duration<double>>(t1-t0);
	printf("CPU FMA execution took %f seconds. \n", t_sum.count()); 	

	t0= high_resolution_clock::now();  
	//Directive to transfer data to GPU
	#pragma acc enter data copyin(A[:rows*cols],B[:rows*cols],C[:rows*cols])
	//OpenACC FMA execution on device
	ACC_FMA(A,B,C,accD,rows,cols); 
	t1 = high_resolution_clock::now();
        
	t_sum = duration_cast<duration<double>>(t1-t0);
        printf("GPU FMA execution with OpenAcc took %f seconds. \n", t_sum.count());

	//Result Verification
        printf("CPU to OpenACC GPU result confirmation: \t");
        MatrixVerification(D,accD,rows,cols,Tol);

	//Display elements to see precision w/o FMA flag
	printf("\nCPU results");
	DisplayElements(D,cols);
	printf("\nGPU acc results:");
	DisplayElements(accD, cols);
	
	//Cleaning up Memory Usage 
	delete[] A; 
	delete[] B; 
	delete[] C; 
	delete[] D; 
	delete[] accD; 
	
	return 0; 
}
