# Compilers and linkers
CC=nvc++
CUDACC=nvcc

# Compilation flags
CFLAGS= -g -O3 -std=c++11 -Wall
CUDACFLAGS= -g -std=c++11 -O3 -I${CUDA_ROOT_PATH}/include

ifeq ($(_OPENACC),true)
ACCFLAGS= -acc -gpu=cc60,cc70,managed -Minfo=all
else 
ACCFLAGS= 
endif

# Linking Flags
LDFLAGS=
CUDALDFLAGS=-L${CUDA_ROOT_PATH}/lib64 -lcudart

# Description of necessary files and binaries
# List of object files separated by spaces (e.g. main.o device.o)
OBJECTS= main.o common.o functions.o FMA.o 
# The binary to be built by this Makefile
BINARY= fma.exe


all: $(BINARY) 

$(BINARY): $(OBJECTS)
	$(CC) -o $(BINARY) $(OBJECTS) $(ACCFLAGS) $(LDFLAGS) $(CUDALDFLAGS) 

%.o: %.cpp Makefile
	$(CC) -c $(CFLAGS) $(ACCFLAGS) $<

%.o: %.cu Makefile
	$(CUDACC) -c $(CUDACFLAGS) $<

clean:
	rm *.o fma.exe

