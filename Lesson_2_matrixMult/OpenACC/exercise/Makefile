# Compilers and linkers
CC=nvc++

# Compilation flags
CFLAGS= -g -O3 -std=c++11 -Wall

ifeq ($(_OPENACC),true)
ACCFLAGS= -acc -gpu=cc60,cc70,managed -Minfo=all -I${NVHPC_ROOT_PATH}/include
else 
ACCFLAGS=
endif

# Description of necessary files and binaries
# List of object files separated by spaces (e.g. main.o device.o)
OBJECTS= main.o common.o functions.o matrix_mult.o 
# The binary to be built by this Makefile
BINARY= matrix_mult.exe


all: $(BINARY) 

$(BINARY): $(OBJECTS)
	$(CC) -o $(BINARY) $(OBJECTS) $(ACCFLAGS) 

%.o: %.cpp Makefile
	$(CC) -c $(CFLAGS) $(ACCFLAGS) $<

%.o: %.cc Makefile
	$(CC) -c $(CFLAGS) $(ACCFLAGS) $<

clean:
	rm *.o $(BINARY)

