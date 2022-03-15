
# Compilers and linkers
CC=nvc++
CUDACC=nvcc

# Compilation flags
CFLAGS= -g -O3 -std=c++11 -Wall
CUDACFLAGS= -g -std=c++11 -O3 -I${CUDA_ROOT_PATH}/include

# Linking Flags
LDFLAGS=
CUDALDFLAGS=-L${CUDA_ROOT_PATH}/lib64 -lcudart


# Description of necessary files and binaries
# List of object files separated by spaces (e.g. main.o device.o)
OBJECTS= main.o functions.o common.o matrixMul.o

# The binary to be built by this Makefile
BINARY= matMult.exe

all: $(BINARY)

$(BINARY): $(OBJECTS)
	$(CC) -o $(BINARY) $(OBJECTS) $(LDFLAGS) $(CUDALDFLAGS) 

%.o: %.cpp Makefile
	$(CC) -c $(CFLAGS)  $<

%.o: %.cu Makefile
	$(CUDACC) -c $(CUDACFLAGS) $<

clean:
	rm *.o $(BINARY)                      
