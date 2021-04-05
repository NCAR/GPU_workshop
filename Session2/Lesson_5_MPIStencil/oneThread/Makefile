# Compilers and linkers
CC=nvc++

# Compilation flags
CFLAGS= -g -O3 -std=c++11 -Wall

ifeq ($(_OPENACC),true)
ACCFLAGS= -acc -ta=tesla,cc60,cc70,managed -Minfo=all -I${NVHPC_ROOT_PATH}/include
else 
ACCFLAGS=
endif

# Description of necessary files and binaries
# List of object files separated by spaces (e.g. main.o device.o)
OBJECTS= main.o functions.o common.o stencil.o 
# The binary to be built by this Makefile
BINARY= acc_stencil.exe


all: $(BINARY) 

$(BINARY): $(OBJECTS)
	$(CC) -o $(BINARY) $(OBJECTS) $(ACCFLAGS) 

%.o: %.cpp Makefile
	$(CC) -c $(CFLAGS) $<

%.o: %.cc Makefile
	$(CC) -c $(CFLAGS) $(ACCFLAGS) $<

clean:
	rm *.o $(BINARY)
