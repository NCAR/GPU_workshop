/* common.cpp
 * To contain functions that are common across all lessons
 */

/* Sets all values in array equal to val */
void InitializeMatrixSame(float *array, const int ny, const int nx, const float val){
	// p serves as another pointer to the start rows within array
	float *p = array;

	for(int i=0; i<ny; i++){
		for(int j=0; j<nx; j++){
			p[j] = val;
		}
		// Advance p to the next row
		p += nx;
	}
}

/* Sets all elements of array to a number between [-1,1] */
void InitializeMatrixRand(float *array, const int ny, const int nx){
	// p serves as another pointer to the start rows within array
	float *p = temp;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			p[j] = ((float)rand() / (RAND_MAX + 1)*(RANGE_MAX - RANGE_MIN) + RANGE_MIN);
		}
		// Advance p to the next row
		p += nx;
	}
}

/* Compares the matrices element-wise and prints an error message if 
 * the difference between values is above fTolerance
 */
void MatrixVerification(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance){
	// Pointers for rows in each matrix
	float *p = hostC;
	float *q = gpuC;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			if (fabs(p[j] - q[j]) > fTolerance)
			{
				cout << "Error: " << fabs(p[j] - q[j]) " > " << fTolerance  << endl;
				cout << "\thostC[" << (i) << "][" << (j + 1) << "] = " << p[j] << endl;
				cout << "\tgpuC[" << (i) << "][" << (j + 1) << "] = " << q[j] << endl;
				return;
			}
		}
		p += nx;
		q += nx;
	}
}

