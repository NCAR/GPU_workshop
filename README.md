# Purpose
For a workshop series presented by the Special Technical Projects Team part of the Computational and Information Systems Laboratory at the National Center for Atmospheric Research

Created in Q1 of 2021 using NVIDIA High Performance Computing Software Development Kit 20.11 and CUDA 11.0.3. The code is written to build and run on UCAR's Casper HPC system.

## Team Members

* Dylan Dickerson - Graduate Reserach Assistant CISL & University of Wyoming
* Evan MacBride - Student Assistant III CISL & University of Delaware
* Krishna Chemudupati - Student Assistant II CISL & University of Colorado Boulder
* Briley James - Student Assisstant III CISL & University of Wyoming
* Oreoluwa Babatunde - Student Assistant II & University of Wyoming
* Supreeth Suresh - ssuresh@ucar.edu - Software Engineer II CISL
* Cena Miller - Software Engineer II CISL


# Instructions
Any code in a leaf directory (e.g. "Lesson_1_matrixAdd/CUDA/solution") can be built using the `build.sh` script to set the environment and run the `make clean` and `make` commands. The executable can be submitted to the job scheduler on UCAR's Casper system (after building) by using `sbatch submit.sh`. The output can be examined within a file of the form `log.*_JobNum.out`

To build:

        ./build.sh

To run:

        sbatch submit.sh

# Contents of A Lesson

Each lesson has two to three sub-directories:
* CUDA - Contains the CUDA version of the code
* OpenACC - Contains the OpenACC version of the same code
* OpenACC_FORTRAN - OpenACC added to a FORTRAN version of the code

Both CUDA and OpenACC directories have at least these two sub-directories:
* Exercise - This is the code that you will be working on.
* Solution - This is the solution code.

Within each sub-directory is a similar set of files:
* build.sh - "./build.sh" will load required modules and then call the make clean and make commands with the Makefile to compile all files and create an executable.
* common.cpp - Includes all the common functions for initializing matrices, printing matrices, and verifying the result against the CPU execution.
* language specific file : It is either .cu extension for CUDA or .cc extension for OpenACC. This file contains the code to be executed on the GPU.
* functions.cpp - Contains the code to be run on the host.
* Makefile - Compiles all files using appropriate flags and creates the desired executable. Differs slightly between CUDA and OpenACC in terms of compiler flags. It can also be used to clean the object and executable files.
* main.cpp - Calls both the host and device routines and calculates the time taken for each.
* pch.h - Header file that contains all function declarations from all other files.
* submit.sh - submit shell script that submits the job. To be used as "sbatch submit.sh".
