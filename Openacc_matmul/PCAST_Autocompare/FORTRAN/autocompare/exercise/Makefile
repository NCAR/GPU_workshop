# Compilers and linkers
F90=nvfortran

# Compilation flags
FFLAGS= -O3 -Wall -Mfree

ACCFLAGS= -acc -ta=tesla:cc70,cuda11.1 -Minfo=accel -I${NVHPC_ROOT_PATH}/include

# Description of necessary files and binaries
# List of object files separated by spaces (e.g. main.o device.o)
OBJECTS= matrix_mult.o 
# The binary to be built by this Makefile
BINARY= matrix_mult.exe


all: $(BINARY) 

$(BINARY): $(OBJECTS)
	$(F90) -o $(BINARY) $(OBJECTS) $(ACCFLAGS) 

%.o: %.f90 Makefile
	$(F90) -c $(FFLAGS) $(ACCFLAGS) $<




clean:
	rm *.o $(BINARY)

