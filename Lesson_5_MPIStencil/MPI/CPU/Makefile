# Compilers and linkers
MPICC=mpic++

# Compilation flags
CFLAGS= -g -O3 -std=c++11 -Wall

# Description of necessary files and binaries
# List of object files separated by spaces (e.g. main.o device.o)
OBJECTS= main.o functions.o common.o
# The binary to be built by this Makefile
BINARY= mpi_cpu_stencil.exe


all: $(BINARY) 

$(BINARY): $(OBJECTS)
	$(MPICC) -o $(BINARY) $(OBJECTS) $(ACCFLAGS) 

%.o: %.cpp Makefile
	$(MPICC) -c $(CFLAGS) $<

%.o: %.cc Makefile
	$(MPICC) -c $(CFLAGS) $(ACCFLAGS) $<

clean:
	rm *.o $(BINARY)
