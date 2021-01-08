# Lesson_Dylan
This lesson will explore the concept of stenciling that has been somewhat established already in the Matrix Multiplication lessons and extends stenciling to a distributed/MPI program. The Jacobi method is an iterative solver for linear systems of equations of the form $A\vec{x}=\vec{b}$ where $A$ is a square matrix N by N, $\vec{x}$ is a vector to be found of length $N$, $\vec{b}$ is also a vector of length $N$. This method is guaranteed to converge for strictly diagonally dominant matrices (the diagonal in each row has a higher magnitude than the sum of all the other magnitudes in the row, or $|A_{ii}| > \sum_{j \neq i}{|A_{ij}|}$ for each row).

# Build Instructions
When logged into Casper and from this folder run

	./build.sh

# Submission Instructions
Again on Casper, submit the build executable to the SLURM scheduler with

	sbatch submit.sh


