#CC=pgc++
#CUDACC=nvcc
#LDFLAGS=
#CFLAGS= -g -O3 -std=c++11 
#CUDALIB=/glade/u/apps/dav/opt/cuda/11.0.3
#CUDAFLAGS= -g -std=c++11 -O3 

#all: output.exe

#output.exe: matrix_mul_solution.o  
#	$(CUDACC)  matrix_mul_solution.o -o output.exe

#%.o: %.cpp makefile
#	$(CC) -c $(CFLAGS) $<

#%.o: %.cu makefile
#	$(CUDACC) -c $(CUDACFLAGS) $<
#
#clean:
#	rm *.o output.exe



# Compilers and linkers
CC=pgc++
CUDACC=nvcc

# Compilation flags
CFLAGS= -g -O3 -std=c++11 -Wall
CUDACFLAGS= -g -std=c++11 -O3 -I${CUDA_ROOT_PATH}/include
#CUDAFLAGS= -g -std=c++11 -O3 


# Linking Flags
LDFLAGS=
CUDALDFLAGS=-L${CUDA_ROOT_PATH}/lib64 -lcudart

# Description of necessary files and binaries
# List of object files separated by spaces (e.g. main.o device.o)
OBJECTS= main.o functions.o common.o matrixMul.o

# The binary to be built by this Makefile
BINARY= output.exe

all: $(BINARY)

$(BINARY): $(OBJECTS)
	$(CC) -o $(BINARY) $(OBJECTS) $(LDFLAGS) $(CUDALDFLAGS) 

%.o: %.cpp Makefile
	$(CC) -c $(CFLAGS) $<

%.o: %.cu Makefile
	$(CUDACC) -c $(CUDACFLAGS) $<

clean:
	rm *.o $(BINARY)                      
