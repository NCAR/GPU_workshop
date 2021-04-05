# Compilers and linkers
CC=nvc++
MPICC=mpic++

# Compilation flags
CFLAGS= -g -O3 -std=c++11 -Wall

ifeq ($(_OPENACC),true)
ACCFLAGS= -acc -ta=tesla,cc60,cc70 -Minfo=accel -I${NVHPC_ROOT_PATH}/include
else 
ACCFLAGS=
endif

# Description of necessary files and binaries
# List of object files separated by spaces (e.g. main.o device.o)
OBJECTS= main.o common.o stencil.o 
# The binary to be built by this Makefile
BINARY= mpi_acc_stencil.exe


all: $(BINARY) 

$(BINARY): $(OBJECTS)
	$(MPICC) -o $(BINARY) $(OBJECTS) $(ACCFLAGS) 

%.o: %.cpp Makefile
	$(MPICC) -c $(CFLAGS) $<

%.o: %.cc Makefile
	$(MPICC) -c $(CFLAGS) $(ACCFLAGS) $<

clean:
	rm *.o $(BINARY)
