
# Compilers and linkers
CC=pgc++
CUDACC=nvcc

# Compilation flags
CFLAGS= -g -O3 -std=c++11 -Wall
CUDACFLAGS= -g -std=c++11 -O3 -I${CUDA_ROOT_PATH}/include

# Linking Flags
LDFLAGS=
CUDALDFLAGS=-L${CUDA_ROOT_PATH}/lib64 -lcudart

ifeq ($(_OPENACC),true)
ACCFLAGS= -acc -gpu=cc60,cc70,managed -Minfo=all -I${NVHPC_ROOT_PATH}/include
else 
ACCFLAGS= 
endif


# Description of necessary files and binaries
# List of object files separated by spaces (e.g. main.o device.o)
OBJECTS= main.o functions.o common.o matrixMul.o 

# The binary to be built by this Makefile
BINARY= output.exe

all: $(BINARY)

$(BINARY): $(OBJECTS)
	$(CC) -o $(BINARY) $(OBJECTS) $(LDFLAGS) $(CUDALDFLAGS) $(ACCFLAGS)

%.o: %.cpp Makefile
	$(CC) -c $(CFLAGS) $(ACCFLAGS)  $<

%.o: %.cc Makefile
	$(CC) -c $(CFLAGS) $(ACCFLAGS) $<

clean:
	rm *.o  $(BINARY)                      
